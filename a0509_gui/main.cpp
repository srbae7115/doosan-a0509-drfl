// A0509 Qt GUI on DRFL (no ROS). Layout follows the UR5e GUI (ur5e-ros2-gui):
//   현재 상태 -> MoveJ | MoveL -> 조그 (X/Y/Z/Roll/Pitch/Yaw ±, J1~J6 ±) -> 동작 진행 -> 정지/속도/도움말 -> 로그
// Control flow (same as a0509_menu):
//   connect -> control access -> auto servo ON (SAFE_OFF only) -> AUTONOMOUS -> MoveJ / MoveL / jog
//   -> on exit: stop if moving, restore previous robot mode, disconnect.
// STOP button / Esc / Space -> MoveStop(QUICK). Software stop does NOT replace the E-stop.
// Feedback (joints + TCP) is used only while the monitoring stream is fresh (< 500 ms); before the first
// sample get_current_posj() returns all zeros, which would otherwise be saved as "home".
// MoveL / Cartesian jog: X/Y/Z [mm] + Roll/Pitch/Yaw [deg] (absolute / relative base / relative tool). The
// target is computed in pose_math.h from the monitored TCP (get_current_posx() hangs on controller 2.8) and
// sent as an absolute base-frame amovel in the controller's ZYZ [A, B, C].
// Build: cd a0509_gui && mkdir build && cd build && cmake .. && make   (controller GF02.x -> DRCF_VERSION=2)
#include "DRFLEx.h"
#include "pose_math.h"

#include <QAbstractSpinBox>
#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QEvent>
#include <QFont>
#include <QFrame>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QKeyEvent>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QShortcut>
#include <QSlider>
#include <QTimer>
#include <QVBoxLayout>
#include <QWidget>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace DRAFramework;
using Clock = std::chrono::steady_clock;
using Pose = std::array<float, 6>;

namespace {
// Conservative software limits [deg] (A0509 spec: J1,J2,J4,J6 +-360, J3 +-150, J5 +-135)
constexpr float LIMIT_DEG[6] = {350.0f, 350.0f, 145.0f, 350.0f, 130.0f, 350.0f};
constexpr float JOINT_TOL_DEG = 0.1f;
constexpr double J_CONFIRM_DEG = 30.0;  // MoveJ larger than this (max joint delta) -> confirmation dialog
// MoveL: "reached" tolerance, and the size above which a confirmation dialog is shown
constexpr double LIN_TOL_MM = 1.0, LIN_TOL_DEG = 0.5;
constexpr double LIN_CONFIRM_MM = 100.0, LIN_CONFIRM_DEG = 30.0;
// MoveL workspace guard: A0509 reach 900 mm from J2 (shoulder, ~155.5 mm above the base).
// The Z floor is a GUI spinbox (default 200 mm): no TCP offset is set on this robot, so posx is the
// flange face and the 2F-85 fingers hang ~150 mm below it.
constexpr double LIN_SHOULDER_Z_MM = 155.5, LIN_MAX_REACH_MM = 850.0;
// Jog step limits
constexpr double JOG_MAX_MM = 50.0, JOG_MAX_DEG = 15.0, JOG_MAX_JDEG = 10.0;

// Progress / state colors (same as the UR5e GUI)
const char* const C_IDLE = "#757575";
const char* const C_RUN = "#1565c0";
const char* const C_OK = "#2e7d32";
const char* const C_STOP = "#ef6c00";
const char* const C_FAIL = "#c62828";

// ---- state shared with DRFL callback threads ----
std::atomic<bool> g_access{false}, g_denied{false}, g_disconnected{false}, g_fault{false};
std::mutex g_mtx;
Pose g_tcp{};                  // TCP in base [mm, ZYZ deg]
float g_rotm[3][3]{};          // TCP rotation matrix from the same sample (ZYZ convention self-check)
Clock::time_point g_tcp_stamp{};
std::vector<std::string> g_logs;

void push_log(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_logs.push_back(s);
}

const char* state_name(int s) {
    static const char* n[] = {"INITIALIZING", "STANDBY", "MOVING", "SAFE_OFF", "TEACHING",
                              "SAFE_STOP", "EMERGENCY_STOP", "HOMMING", "RECOVERY",
                              "SAFE_STOP2", "SAFE_OFF2"};
    return (s >= 0 && s <= STATE_SAFE_OFF2) ? n[s] : "OTHER";
}

bool is_fault(ROBOT_STATE s) {
    return s == STATE_EMERGENCY_STOP || s == STATE_SAFE_STOP || s == STATE_SAFE_STOP2;
}

void on_access(const MONITORING_ACCESS_CONTROL a) {
    if (a == MONITORING_ACCESS_CONTROL_GRANT) { g_access = true; push_log("제어권 획득"); }
    if (a == MONITORING_ACCESS_CONTROL_LOSS)  { g_access = false; push_log("제어권 상실"); }
    if (a == MONITORING_ACCESS_CONTROL_DENY)  { g_denied = true; push_log("제어권 요청 거부됨"); }
}
void on_state(const ROBOT_STATE s) {
    if (is_fault(s)) g_fault = true;
    push_log(std::string("상태 -> ") + state_name(s));
}
void on_alarm(LPLOG_ALARM a) {
    push_log("[알람] level=" + std::to_string(a->_iLevel) + " group=" + std::to_string(a->_iGroup) +
             " index=" + std::to_string(a->_iIndex) + " " + a->_szParam[0]);
}
void on_disconnected() {
    g_disconnected = true;
    g_access = false;
    push_log("컨트롤러 연결 끊김");
}
void on_monitoring(const LPMONITORING_DATA_EX d) {
    std::lock_guard<std::mutex> lk(g_mtx);
    for (int i = 0; i < 6; ++i) g_tcp[i] = d->_tCtrl._tTask._fActualPos[0][i];
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) g_rotm[r][c] = d->_tCtrl._tTask._fRotationMatrix[r][c];
    g_tcp_stamp = Clock::now();
}
// Before the first monitoring sample get_current_posj() returns all zeros -> never trust stale data.
bool fresh_locked() {  // caller holds g_mtx
    return g_tcp_stamp != Clock::time_point{} && Clock::now() - g_tcp_stamp < std::chrono::milliseconds(500);
}
bool monitoring_fresh() {
    std::lock_guard<std::mutex> lk(g_mtx);
    return fresh_locked();
}
// TCP [mm, ZYZ deg] from the monitoring stream; false (p untouched) when stale or invalid.
bool read_tcp(Pose& p) {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!fresh_locked()) return false;
    for (float v : g_tcp)
        if (!std::isfinite(v)) return false;
    p = g_tcp;
    return true;
}
}  // namespace

class MainWindow : public QWidget {
public:
    MainWindow() {
        setWindowTitle("A0509 MoveJ / MoveL (DRFL)");
        auto* root = new QGridLayout(this);
        root->setVerticalSpacing(4);
        QFont mono("Monospace");
        mono.setStyleHint(QFont::TypeWriter);

        // ---------- 연결 ----------
        auto* connRow = new QHBoxLayout;
        connRow->addWidget(new QLabel("IP"));
        ip_ = new QLineEdit("192.168.137.101");
        ip_->setMaximumWidth(150);
        connRow->addWidget(ip_);
        force_ = new QCheckBox("강제 제어권 요청 (TP 승인 없이)");
        connRow->addWidget(force_);
        connectBtn_ = makeButton("연결");
        connRow->addWidget(connectBtn_);
        connRow->addStretch();
        root->addLayout(connRow, 0, 0, 1, 2);

        // ---------- 현재 상태 ----------
        auto* st = new QGroupBox("현재 상태");
        auto* sg = new QGridLayout(st); sg->setVerticalSpacing(3);
        static const char* const kJ[6] = {"J1 [°]", "J2 [°]", "J3 [°]", "J4 [°]", "J5 [°]", "J6 [°]"};
        static const char* const kP[6] = {"X [mm]", "Y [mm]", "Z [mm]", "Roll [°]", "Pitch [°]", "Yaw [°]"};
        for (int i = 0; i < 6; ++i) {
            sg->addWidget(centered(new QLabel(kJ[i])), 0, i);
            curJ_[i] = valueLabel(mono);
            sg->addWidget(curJ_[i], 1, i);
            sg->addWidget(centered(new QLabel(kP[i])), 2, i);
            curP_[i] = valueLabel(mono);
            sg->addWidget(curP_[i], 3, i);
        }
        zyzLbl_ = new QLabel("TP 표시 (ZYZ): -");
        zyzLbl_->setToolTip("TP와 컨트롤러는 자세를 ZYZ 오일러(A, B, C)로 표시합니다. X/Y/Z는 플랜지 면 기준 (TCP 오프셋 없음)");
        auto* pw = new QHBoxLayout;
        connLbl_ = badge(); stateLbl_ = badge(); modeLbl_ = badge(); accessLbl_ = badge();
        pw->addWidget(new QLabel("연결:")); pw->addWidget(connLbl_);
        pw->addWidget(new QLabel("  상태:")); pw->addWidget(stateLbl_);
        pw->addWidget(new QLabel("  모드:")); pw->addWidget(modeLbl_);
        pw->addWidget(new QLabel("  제어권:")); pw->addWidget(accessLbl_);
        servoBtn_ = makeButton("Servo ON");
        servoBtn_->setStyleSheet("QPushButton{font-weight:bold;}");
        pw->addSpacing(12);
        pw->addWidget(servoBtn_);
        pw->addSpacing(16);
        pw->addWidget(zyzLbl_);
        pw->addStretch();
        sg->addLayout(pw, 4, 0, 1, 6);
        root->addWidget(st, 1, 0, 1, 2);

        // ---------- MoveJ | MoveL ----------
        root->addWidget(buildMoveJBox(), 2, 0);
        root->addWidget(buildMoveLBox(), 2, 1);

        // ---------- 조그 ----------
        root->addWidget(buildJogBox(), 3, 0, 1, 2);

        // ---------- 동작 진행 ----------
        auto* fp = new QGroupBox("동작 진행");
        auto* fl = new QHBoxLayout(fp);
        mvState_ = new QLabel;
        mvState_->setMinimumWidth(130);
        mvState_->setAlignment(Qt::AlignCenter);
        mvBar_ = new QProgressBar;
        mvBar_->setRange(0, 100);
        mvBar_->setMinimumWidth(300);
        mvInfo_ = new QLabel;
        mvInfo_->setMinimumWidth(200);
        mvInfo_->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);  // long text never resizes the window
        fl->addWidget(mvState_);
        fl->addWidget(mvBar_);
        fl->addWidget(mvInfo_, 1);
        root->addWidget(fp, 4, 0, 1, 2);
        mvShow("대기", C_IDLE, 0, "");

        // ---------- 하단 ----------
        auto* fc = new QHBoxLayout;
        stopBtn_ = makeButton("■ 정지 (Esc / Space)");
        stopBtn_->setMinimumHeight(46);
        stopBtn_->setMinimumWidth(220);
        stopBtn_->setStyleSheet("QPushButton{background:#c62828;color:white;font-weight:bold;font-size:16px;}"
                                "QPushButton:disabled{background:#9e9e9e;}");
        fc->addWidget(stopBtn_);
        fc->addSpacing(20);
        fc->addWidget(new QLabel("전체 속도"));
        speed_ = new QSlider(Qt::Horizontal);
        speed_->setRange(1, 100);
        speed_->setValue(100);
        speed_->setMinimumWidth(180);
        speed_->setFocusPolicy(Qt::NoFocus);
        speed_->setToolTip("change_operation_speed: 모든 이동 속도에 곱해지는 비율. 컨트롤러에 유지되며 TP 속도와 같은 값입니다");
        speedLbl_ = new QLabel("100%");
        speedLbl_->setMinimumWidth(40);
        speedBtn_ = makeButton("적용");
        fc->addWidget(speed_);
        fc->addWidget(speedLbl_);
        fc->addWidget(speedBtn_);
        fc->addStretch();
        helpBtn_ = makeButton("❓ 도움말 (F1)");
        fc->addWidget(helpBtn_);
        root->addLayout(fc, 5, 0, 1, 2);

        log_ = new QPlainTextEdit;
        log_->setReadOnly(true);
        log_->setMaximumBlockCount(500);
        log_->setFocusPolicy(Qt::NoFocus);
        log_->setFont(mono);
        log_->setMinimumHeight(70);
        root->addWidget(log_, 6, 0, 1, 2);
        root->setColumnStretch(0, 1);
        root->setColumnStretch(1, 1);
        root->setRowStretch(6, 1);

        QObject::connect(connectBtn_, &QPushButton::clicked, [this] { robot_ ? disconnectRobot() : connectRobot(); });
        QObject::connect(servoBtn_, &QPushButton::clicked, [this] { beginServo(); });
        QObject::connect(stopBtn_, &QPushButton::clicked, [this] { stopNow("정지 버튼"); });
        QObject::connect(speed_, &QSlider::valueChanged, [this](int v) { speedLbl_->setText(QString("%1%").arg(v)); });
        QObject::connect(speedBtn_, &QPushButton::clicked, [this] { applySpeed(); });
        QObject::connect(helpBtn_, &QPushButton::clicked, [this] { showHelp(); });
        for (auto key : {Qt::Key_Escape, Qt::Key_Space}) {
            auto* sc = new QShortcut(QKeySequence(key), this);
            sc->setContext(Qt::ApplicationShortcut);
            QObject::connect(sc, &QShortcut::activated, [this] { stopNow("키보드 정지"); });
        }
        auto* f1 = new QShortcut(QKeySequence(Qt::Key_F1), this);
        QObject::connect(f1, &QShortcut::activated, [this] { showHelp(); });
        // A focused text field would otherwise claim Space/Esc and the stop shortcut would not fire while
        // the cursor is in it -> see eventFilter().
        QCoreApplication::instance()->installEventFilter(this);

        QObject::connect(&timer_, &QTimer::timeout, [this] { tick(); });
        timer_.start(100);
        refreshUi();
        resize(1180, 1050);
    }
    ~MainWindow() override { QCoreApplication::instance()->removeEventFilter(this); }

protected:
    void closeEvent(QCloseEvent* e) override {
        disconnectRobot();
        e->accept();
    }

    bool eventFilter(QObject* obj, QEvent* e) override {
        if (e->type() == QEvent::ShortcutOverride) {
            const auto* k = static_cast<QKeyEvent*>(e);
            // Leave the override un-accepted for every receiver: no widget can take the key, so the
            // Esc / Space STOP shortcut always fires (and nothing is typed into the field).
            if ((k->key() == Qt::Key_Escape || k->key() == Qt::Key_Space) && k->modifiers() == Qt::NoModifier)
                return true;
        } else if (e->type() == QEvent::Wheel) {
            // A mouse wheel passing over a number field / step box must not silently change a target.
            auto isInput = [](QObject* o) {
                return o && (qobject_cast<QAbstractSpinBox*>(o) || qobject_cast<QComboBox*>(o));
            };
            if (isInput(obj) || isInput(obj->parent())) return true;
        }
        return QWidget::eventFilter(obj, e);
    }

private:
    enum class Phase { Off, WaitAccess, WaitServo, Ready, Moving };
    enum class MoveKind { Joint, Linear };

    // ---------- widget helpers ----------
    QPushButton* makeButton(const QString& text) {
        auto* b = new QPushButton(text);
        b->setFocusPolicy(Qt::NoFocus);  // Space must never "press" a focused motion button
        return b;
    }

    static QDoubleSpinBox* makeSpin(double lo, double hi, double value, int decimals, double step = 1.0,
                                    int minWidth = 110) {
        auto* s = new QDoubleSpinBox;
        s->setRange(lo, hi);
        s->setDecimals(decimals);
        s->setSingleStep(step);
        s->setValue(value);
        s->setButtonSymbols(QAbstractSpinBox::NoButtons);
        s->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        s->setMinimumWidth(minWidth);
        s->setMaximumHeight(26);
        return s;
    }

    static QComboBox* makeStepCombo(const QStringList& values, const QString& def) {
        auto* c = new QComboBox;
        c->setEditable(true);
        c->addItems(values);
        c->setCurrentText(def);
        c->setMinimumWidth(70);
        return c;
    }

    static QLabel* centered(QLabel* l) {
        l->setAlignment(Qt::AlignCenter);
        return l;
    }

    static QLabel* valueLabel(const QFont& f) {
        auto* l = new QLabel("-");
        l->setFont(f);
        l->setFrameStyle(QFrame::Panel | QFrame::Sunken);
        l->setAlignment(Qt::AlignCenter);
        l->setMinimumWidth(100);
        return l;
    }

    static QLabel* badge() {
        auto* l = new QLabel("-");
        l->setFrameStyle(QFrame::Panel | QFrame::Raised);
        l->setAlignment(Qt::AlignCenter);
        l->setMinimumWidth(110);
        return l;
    }

    static void setBadge(QLabel* l, const QString& text, const char* bg) {
        l->setText(text);
        l->setStyleSheet(bg ? QString("QLabel{background:%1;color:white;font-weight:bold;padding:2px 6px;}").arg(bg)
                            : QString("QLabel{font-weight:bold;padding:2px 6px;}"));
    }

    static QRadioButton* radio(const QString& text, const QString& tip) {
        auto* r = new QRadioButton(text);
        r->setFocusPolicy(Qt::NoFocus);
        r->setToolTip(tip);
        return r;
    }

    // Number from an editable step combo box, rejected unless 0 < value <= hi.
    bool comboValue(QComboBox* c, double hi, double& out, const QString& what) {
        bool ok = false;
        const double v = c->currentText().trimmed().toDouble(&ok);
        if (!ok || !(v > 0) || v > hi) {
            log(QString("[오류] %1 값 확인: '%2' (0 초과 %3 이하)").arg(what).arg(c->currentText()).arg(hi));
            return false;
        }
        out = v;
        return true;
    }

    // A half-typed field ("", "-") keeps its previous value() -> never move to a number not on screen.
    bool inputsOk(std::initializer_list<QDoubleSpinBox*> boxes, const QString& what) {
        for (auto* s : boxes)
            if (!s->hasAcceptableInput()) {
                log(QString("%1 불가: 입력값을 확인하세요 ('%2')").arg(what).arg(s->text()));
                return false;
            }
        return true;
    }

    // ---------- MoveJ panel ----------
    QGroupBox* buildMoveJBox() {
        auto* box = new QGroupBox("MoveJ  (관절 공간)");
        auto* g = new QGridLayout(box); g->setVerticalSpacing(3);
        for (int i = 0; i < 6; ++i) {
            g->addWidget(new QLabel(QString("J%1 [°]").arg(i + 1)), i, 0);
            jIn_[i] = makeSpin(-360, 360, 0, 3);
            g->addWidget(jIn_[i], i, 1, 1, 3);
        }
        auto* modeRow = new QHBoxLayout;
        jAbs_ = radio("절대", "입력한 관절각 자체가 목표");
        jRel_ = radio("상대(증분)", "현재 관절각에 입력값을 더한 값이 목표");
        auto* grp = new QButtonGroup(box);
        for (auto* r : {jAbs_, jRel_}) { grp->addButton(r); modeRow->addWidget(r); }
        jRel_->setChecked(true);  // all-zero inputs + relative = no motion (safe default)
        modeRow->addStretch();
        g->addLayout(modeRow, 6, 0, 1, 4);
        jVel_ = makeSpin(0.5, 30, 10, 1);
        jAcc_ = makeSpin(1, 60, 20, 1);
        g->addWidget(new QLabel("속도 v [°/s]"), 7, 0);
        g->addWidget(jVel_, 7, 1);
        g->addWidget(new QLabel("가속 a [°/s²]"), 8, 0);
        g->addWidget(jAcc_, 8, 1);

        auto* b1 = new QHBoxLayout;
        jLoadBtn_ = makeButton("현재값 불러오기");
        jHomeValBtn_ = makeButton("원위치 값");
        homeBtn_ = makeButton("⌂ 원위치로 이동");
        setHomeBtn_ = makeButton("원위치 저장");
        jGoBtn_ = makeButton("MoveJ 실행");
        jGoBtn_->setMinimumHeight(36);
        jGoBtn_->setStyleSheet("QPushButton{font-weight:bold;}");
        jLoadBtn_->setToolTip("현재 관절각을 입력칸에 채우고 [절대]로 전환");
        jHomeValBtn_->setToolTip("저장된 원위치 관절각을 입력칸에 채우고 [절대]로 전환 (이동하지 않음)");
        homeBtn_->setToolTip("저장된 원위치로 MoveJ (위 속도/가속 사용)");
        setHomeBtn_->setToolTip("현재 관절 자세를 원위치로 저장 (연결 후 처음 준비될 때 자동 저장됨)");
        jGoBtn_->setToolTip(QString("amovej. 관절 변화가 %1°보다 크면 확인 창이 뜹니다").arg(J_CONFIRM_DEG));
        b1->addWidget(jLoadBtn_);
        b1->addWidget(jHomeValBtn_);
        b1->addWidget(homeBtn_);
        b1->addWidget(setHomeBtn_);
        b1->addStretch();
        b1->addWidget(jGoBtn_);
        g->addLayout(b1, 9, 0, 1, 4);
        g->setRowStretch(10, 1);

        QObject::connect(jLoadBtn_, &QPushButton::clicked, [this] {
            Pose p;
            if (!readJoints(p)) { log("현재값 불러오기 실패: 관절 피드백이 최신이 아닙니다"); return; }
            for (int i = 0; i < 6; ++i) jIn_[i]->setValue(p[i]);
            jAbs_->setChecked(true);
            log("MoveJ 입력 <- 현재 관절 " + poseStr(p));
        });
        QObject::connect(jHomeValBtn_, &QPushButton::clicked, [this] {
            if (!haveHome_) { log("저장된 원위치가 없습니다"); return; }
            for (int i = 0; i < 6; ++i) jIn_[i]->setValue(home_[i]);
            jAbs_->setChecked(true);
            log("MoveJ 입력 <- 원위치 " + poseStr(home_));
        });
        QObject::connect(homeBtn_, &QPushButton::clicked, [this] {
            if (!haveHome_) return;
            if (!inputsOk({jVel_, jAcc_}, "원위치 이동")) return;
            startJoint(home_, "⌂ 원위치", jVel_->value(), jAcc_->value(), true);
        });
        QObject::connect(setHomeBtn_, &QPushButton::clicked, [this] {
            Pose p;
            if (!readJoints(p)) { log("원위치 저장 실패: 관절 피드백이 최신이 아닙니다"); return; }
            home_ = p; haveHome_ = true; log("원위치 저장: " + poseStr(home_));
        });
        QObject::connect(jGoBtn_, &QPushButton::clicked, [this] { doMoveJ(); });
        return box;
    }

    // ---------- MoveL panel ----------
    QGroupBox* buildMoveLBox() {
        auto* box = new QGroupBox("MoveL  (직선, base 좌표계 / TCP)");
        auto* g = new QGridLayout(box); g->setVerticalSpacing(3);
        static const char* const kLabels[6] = {"X [mm]", "Y [mm]", "Z [mm]",
                                               "Roll (Rx) [°]", "Pitch (Ry) [°]", "Yaw (Rz) [°]"};
        static const char* const kTips[6] = {
            "절대: 목표 X / 상대: X 방향 이동량", "절대: 목표 Y / 상대: Y 방향 이동량",
            "절대: 목표 Z / 상대: Z 방향 이동량 (base 기준 + = 위쪽, tool 기준 + = 툴이 바라보는 방향)",
            "X축 기준 회전 (RPY: R = Rz(yaw)·Ry(pitch)·Rx(roll), 고정축 X→Y→Z, ROS rpy와 동일)",
            "Y축 기준 회전", "Z축 기준 회전"};
        for (int i = 0; i < 6; ++i) {
            g->addWidget(new QLabel(kLabels[i]), i, 0);
            lin_[i] = i < 3 ? makeSpin(-2000, 2000, 0, 2) : makeSpin(-360, 360, 0, 3);
            lin_[i]->setToolTip(kTips[i]);
            g->addWidget(lin_[i], i, 1, 1, 3);
        }

        auto* modeRow = new QHBoxLayout;
        linAbs_ = radio("절대", "입력한 X/Y/Z, Roll/Pitch/Yaw 자체가 목표 자세 (base 좌표계)");
        linRelBase_ = radio("상대(base)", "현재 TCP에서 base 좌표축 방향으로 이동 / TCP 위치를 고정한 채 base 축 기준 회전");
        linRelTool_ = radio("상대(tool)", "현재 TCP에서 툴 좌표축 방향으로 이동/회전. 예) Z에 10 → 툴이 바라보는 방향으로 10mm");
        auto* group = new QButtonGroup(box);
        for (auto* r : {linAbs_, linRelBase_, linRelTool_}) { group->addButton(r); modeRow->addWidget(r); }
        linRelBase_->setChecked(true);  // all-zero inputs + relative = no motion (safe default)
        modeRow->addStretch();
        g->addLayout(modeRow, 6, 0, 1, 4);

        linVel_ = makeSpin(1, 250, 20, 1);
        linAcc_ = makeSpin(10, 1000, 100, 0, 10);
        linRotVel_ = makeSpin(1, 60, 10, 1);
        linRotAcc_ = makeSpin(1, 180, 30, 1);
        linVel_->setToolTip("TCP 직선 속도 (amovel vel[0])");
        linAcc_->setToolTip("TCP 직선 가속도 (amovel acc[0])");
        linRotVel_->setToolTip("TCP 회전 속도 (amovel vel[1]). 자세만 바꿀 때는 이 값이 이동 시간을 정합니다");
        linRotAcc_->setToolTip("TCP 회전 가속도 (amovel acc[1])");
        g->addWidget(new QLabel("속도 v [mm/s]"), 7, 0);
        g->addWidget(linVel_, 7, 1);
        g->addWidget(new QLabel("회전 [°/s]"), 7, 2);
        g->addWidget(linRotVel_, 7, 3);
        g->addWidget(new QLabel("가속 a [mm/s²]"), 8, 0);
        g->addWidget(linAcc_, 8, 1);
        g->addWidget(new QLabel("회전 [°/s²]"), 8, 2);
        g->addWidget(linRotAcc_, 8, 3);
        linZMin_ = makeSpin(-500, 1000, 200, 0, 10);
        linZMin_->setToolTip("목표 Z가 이 값보다 낮으면 실행 거부 (MoveL과 직교 조그 모두). TCP가 플랜지면 그리퍼 길이만큼 여유를 두세요");
        g->addWidget(new QLabel("Z 하한 [mm]"), 9, 0);
        g->addWidget(linZMin_, 9, 1);

        auto* btnRow = new QHBoxLayout;
        linLoadBtn_ = makeButton("현재값 불러오기");
        linZeroBtn_ = makeButton("0으로");
        linGoBtn_ = makeButton("MoveL 실행");
        linLoadBtn_->setToolTip("현재 TCP를 X/Y/Z, Roll/Pitch/Yaw로 채우고 [절대]로 전환. 여기서 조금만 고쳐서 실행하면 안전합니다");
        linZeroBtn_->setToolTip("입력칸을 모두 0으로 ([절대]였으면 [상대(base)]로 전환). 상대 이동할 때 움직일 축만 입력하세요");
        linGoBtn_->setToolTip(QString("amovel: TCP가 직선으로 이동합니다. 특이점 근처에서는 실패할 수 있으니 짧은 거리부터 시험하세요.\n"
                                      "%1mm 또는 %2° 보다 큰 이동은 확인 창이 뜹니다. 정지: Esc / Space / 정지 버튼")
                                  .arg(LIN_CONFIRM_MM).arg(LIN_CONFIRM_DEG));
        linGoBtn_->setMinimumHeight(36);
        linGoBtn_->setStyleSheet("QPushButton{font-weight:bold;}");
        btnRow->addWidget(linLoadBtn_);
        btnRow->addWidget(linZeroBtn_);
        btnRow->addStretch();
        btnRow->addWidget(linGoBtn_);
        g->addLayout(btnRow, 10, 0, 1, 4);
        g->setRowStretch(11, 1);

        QObject::connect(linLoadBtn_, &QPushButton::clicked, [this] { loadCurrentTcp(); });
        QObject::connect(linZeroBtn_, &QPushButton::clicked, [this] {
            for (auto* s : lin_) s->setValue(0);
            if (linAbs_->isChecked()) {
                linRelBase_->setChecked(true);
                log("MoveL 입력 0으로 (모드: 절대 -> 상대(base))");
            }
        });
        QObject::connect(linGoBtn_, &QPushButton::clicked, [this] { doMoveL(); });
        return box;
    }

    // ---------- Jog panel ----------
    QGroupBox* buildJogBox() {
        auto* box = new QGroupBox("조그  (버튼 1회 = 1스텝 이동)");
        auto* g = new QGridLayout(box); g->setVerticalSpacing(3);

        auto* opt = new QHBoxLayout;
        opt->addWidget(new QLabel("기준:"));
        jogBase_ = radio("Base", "base 좌표축 방향으로 이동 / TCP 위치를 고정한 채 base 축 기준 회전");
        jogTool_ = radio("Tool", "툴 좌표축 방향으로 이동/회전 (툴이 아래를 볼 때 Z+ = 아래로)");
        auto* grp = new QButtonGroup(box);
        for (auto* r : {jogBase_, jogTool_}) { grp->addButton(r); opt->addWidget(r); }
        jogBase_->setChecked(true);
        opt->addSpacing(12);
        opt->addWidget(new QLabel("이동 스텝 [mm]"));
        jogMm_ = makeStepCombo({"0.5", "1", "5", "10", "20", "50"}, "5");
        opt->addWidget(jogMm_);
        opt->addWidget(new QLabel("  회전 스텝 [°]"));
        jogDeg_ = makeStepCombo({"0.5", "1", "2", "5", "10", "15"}, "2");
        opt->addWidget(jogDeg_);
        opt->addWidget(new QLabel("  속도 [mm/s]"));
        jogV_ = makeSpin(1, 100, 20, 1, 1, 60);
        opt->addWidget(jogV_);
        opt->addWidget(new QLabel("  회전속도 [°/s]"));
        jogW_ = makeSpin(1, 30, 10, 1, 1, 60);
        opt->addWidget(jogW_);
        opt->addStretch();
        g->addLayout(opt, 0, 0, 1, 6);

        static const char* const kC[6] = {"X", "Y", "Z", "Roll", "Pitch", "Yaw"};
        for (int i = 0; i < 6; ++i) {
            cPlus_[i] = makeButton(QString("%1 +").arg(kC[i]));
            cMinus_[i] = makeButton(QString("%1 −").arg(kC[i]));
            for (auto* b : {cPlus_[i], cMinus_[i]}) b->setMinimumHeight(34);
            g->addWidget(cPlus_[i], 1, i);
            g->addWidget(cMinus_[i], 2, i);
            QObject::connect(cPlus_[i], &QPushButton::clicked, [this, i] { jogCart(i, +1); });
            QObject::connect(cMinus_[i], &QPushButton::clicked, [this, i] { jogCart(i, -1); });
        }

        auto* opt2 = new QHBoxLayout;
        opt2->addWidget(new QLabel("관절 스텝 [°]"));
        jogJDeg_ = makeStepCombo({"0.5", "1", "2", "5", "10"}, "1");
        opt2->addWidget(jogJDeg_);
        opt2->addWidget(new QLabel("  관절 속도 [°/s] (가속도 동일)"));
        jogJV_ = makeSpin(0.5, 10, 2, 1, 0.5, 60);
        opt2->addWidget(jogJV_);
        opt2->addStretch();
        g->addLayout(opt2, 3, 0, 1, 6);
        for (int j = 0; j < 6; ++j) {
            jPlus_[j] = makeButton(QString("J%1 +").arg(j + 1));
            jMinus_[j] = makeButton(QString("J%1 −").arg(j + 1));
            for (auto* b : {jPlus_[j], jMinus_[j]}) b->setMinimumHeight(34);
            g->addWidget(jPlus_[j], 4, j);
            g->addWidget(jMinus_[j], 5, j);
            QObject::connect(jPlus_[j], &QPushButton::clicked, [this, j] { jogJoint(j, +1); });
            QObject::connect(jMinus_[j], &QPushButton::clicked, [this, j] { jogJoint(j, -1); });
        }
        return box;
    }

    pm::LinMode linMode() const {
        if (linAbs_->isChecked()) return pm::LinMode::Absolute;
        return linRelTool_->isChecked() ? pm::LinMode::RelTool : pm::LinMode::RelBase;
    }

    static QString modeName(pm::LinMode m) {
        return m == pm::LinMode::Absolute ? "절대" : m == pm::LinMode::RelBase ? "상대(base)" : "상대(tool)";
    }

    // "[x, y, z | roll, pitch, yaw]" (or ZYZ A, B, C)
    static QString vecStr(const pm::Vec6& v) {
        return QString("[%1, %2, %3 | %4, %5, %6]")
            .arg(v[0], 0, 'f', 2).arg(v[1], 0, 'f', 2).arg(v[2], 0, 'f', 2)
            .arg(v[3], 0, 'f', 3).arg(v[4], 0, 'f', 3).arg(v[5], 0, 'f', 3);
    }

    void loadCurrentTcp() {
        Pose tcp;
        if (!read_tcp(tcp)) { log("현재값 불러오기 실패: TCP 피드백이 최신이 아닙니다"); return; }
        const pm::Vec6 v = pm::xyzrpy_from_frame(pm::frame_from_posx(tcp));
        for (int i = 0; i < 6; ++i) lin_[i]->setValue(v[i]);
        linAbs_->setChecked(true);
        log("MoveL 입력 <- 현재 TCP [X,Y,Z | R,P,Y] " + vecStr(v));
    }

    // ---------- motion commands ----------
    void doMoveJ() {
        if (!robot_) return;
        if (!inputsOk({jIn_[0], jIn_[1], jIn_[2], jIn_[3], jIn_[4], jIn_[5], jVel_, jAcc_}, "MoveJ")) return;
        Pose cur;
        if (!readJoints(cur)) { log("MoveJ 불가: 관절 피드백이 최신이 아닙니다"); return; }
        const bool abs = jAbs_->isChecked();
        Pose target;
        for (int i = 0; i < 6; ++i) target[i] = static_cast<float>(abs ? jIn_[i]->value() : cur[i] + jIn_[i]->value());
        startJoint(target, QString("MoveJ %1").arg(abs ? "절대" : "상대"), jVel_->value(), jAcc_->value(), true);
    }

    void doMoveL() {
        if (!robot_) return;
        if (!inputsOk({lin_[0], lin_[1], lin_[2], lin_[3], lin_[4], lin_[5], linVel_, linAcc_, linRotVel_, linRotAcc_,
                       linZMin_}, "MoveL"))
            return;
        Pose tcp;
        if (!read_tcp(tcp)) { log("MoveL 불가: TCP 피드백이 최신이 아닙니다"); return; }
        const pm::LinMode mode = linMode();
        pm::Vec6 in;
        for (int i = 0; i < 6; ++i) in[i] = lin_[i]->value();
        const pm::Frame cur = pm::frame_from_posx(tcp);
        startLinear(tcp, pm::movel_target(cur, mode, in), "MoveL " + modeName(mode), linVel_->value(),
                    linAcc_->value(), linRotVel_->value(), linRotAcc_->value(), true);
    }

    void jogCart(int axis, int sign) {
        if (!robot_) return;
        if (!inputsOk({jogV_, jogW_, linZMin_}, "조그")) return;
        double step;
        if (axis < 3 ? !comboValue(jogMm_, JOG_MAX_MM, step, "이동 스텝") : !comboValue(jogDeg_, JOG_MAX_DEG, step, "회전 스텝"))
            return;
        Pose tcp;
        if (!read_tcp(tcp)) { log("조그 불가: TCP 피드백이 최신이 아닙니다"); return; }
        pm::Vec6 d{};
        d[axis] = sign * step;
        const bool tool = jogTool_->isChecked();
        const pm::Frame cur = pm::frame_from_posx(tcp);
        static const char* const kC[6] = {"X", "Y", "Z", "Roll", "Pitch", "Yaw"};
        const QString label = QString("조그 %1 %2%3%4 (%5)")
                                  .arg(kC[axis]).arg(sign > 0 ? "+" : "−").arg(step).arg(axis < 3 ? "mm" : "°")
                                  .arg(tool ? "Tool" : "Base");
        const double v = jogV_->value(), w = jogW_->value();
        startLinear(tcp, pm::movel_target(cur, tool ? pm::LinMode::RelTool : pm::LinMode::RelBase, d), label, v,
                    std::max(10.0, 3 * v), w, 3 * w, false);
    }

    void jogJoint(int j, int sign) {
        if (!robot_) return;
        if (!inputsOk({jogJV_}, "조그")) return;
        double step;
        if (!comboValue(jogJDeg_, JOG_MAX_JDEG, step, "관절 스텝")) return;
        Pose target;
        if (!readJoints(target)) { log("이동 불가: 관절 피드백이 최신이 아닙니다"); return; }
        target[j] += sign * static_cast<float>(step);
        const double v = jogJV_->value();
        startJoint(target, QString("조그 J%1 %2%3°").arg(j + 1).arg(sign > 0 ? "+" : "−").arg(step), v, v, false);
    }

    // Common joint-space move: limits, optional confirmation, amovej, start tracking.
    void startJoint(const Pose& target, const QString& label, double v, double a, bool confirmBig) {
        QString why;
        if (!robot_ || !canMove(&why)) { log("이동 불가: " + why); return; }
        Pose cur;
        if (!readJoints(cur)) { log("이동 불가: 관절 피드백이 최신이 아닙니다"); return; }
        for (int i = 0; i < 6; ++i)
            if (std::fabs(target[i]) > LIMIT_DEG[i]) {
                log(QString("이동 거부: J%1 목표 %2°가 한계 ±%3°를 넘습니다").arg(i + 1).arg(target[i]).arg(LIMIT_DEG[i]));
                return;
            }
        double dist = 0;
        for (int i = 0; i < 6; ++i) dist = std::max(dist, double(std::fabs(target[i] - cur[i])));
        if (dist < 0.001) { log("이미 목표 위치입니다"); return; }

        if (confirmBig && dist > J_CONFIRM_DEG) {
            const auto ans = QMessageBox::question(
                this, "MoveJ 확인",
                QString("큰 관절 이동입니다: 최대 %1°  (%2)\n\n현재: %3\n목표: %4\n\n"
                        "경로에 사람/장애물이 없는지 확인하세요. 실행할까요?")
                    .arg(dist, 0, 'f', 1).arg(label).arg(poseStr(cur)).arg(poseStr(target)),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (ans != QMessageBox::Yes) { log("MoveJ 취소"); return; }
            // The dialog ran the event loop: the connection or robot state may have changed meanwhile.
            if (!robot_ || !canMove(&why)) { log("이동 불가: " + why); return; }
            Pose now;
            if (!readJoints(now)) { log("이동 불가: 관절 피드백이 최신이 아닙니다"); return; }
            for (int i = 0; i < 6; ++i)
                if (std::fabs(now[i] - cur[i]) > 0.2f) { log("MoveJ 취소: 확인하는 동안 자세가 바뀌었습니다. 다시 실행하세요"); return; }
        }

        Pose t = target;
        g_fault = false;
        if (!robot_->amovej(t.data(), static_cast<float>(v), static_cast<float>(a), 0.f, MOVE_MODE_ABSOLUTE)) {
            log("컨트롤러가 이동 명령을 거부했습니다");
            return;
        }
        moveKind_ = MoveKind::Joint;
        targetJ_ = t;
        totJ_ = dist;
        maxSec_ = pm::trapezoid_time(dist, v, a) + 10.0;
        beginTracking(label);
        log(QString("이동 시작: %1 (%2°/s) -> %3").arg(label).arg(v).arg(poseStr(t)));
    }

    // Common Cartesian move: workspace guard, optional confirmation, amovel (ABS, BASE), start tracking.
    // tcp = the monitored posx the target was computed from.
    void startLinear(const Pose& tcp, const pm::Frame& tgt, const QString& label, double v, double a, double w,
                     double aw, bool confirmBig) {
        QString why;
        if (!robot_ || !canMove(&why)) { log("이동 불가: " + why); return; }
        const pm::Frame cur = pm::frame_from_posx(tcp);
        const double dist = pm::dist(cur.p, tgt.p), ang = pm::rot_angle_deg(cur.R, tgt.R);
        if (dist < 0.01 && ang < 0.01) { log("이미 목표 위치입니다"); return; }
        if (tgt.p[2] < linZMin_->value()) {
            log(QString("이동 거부: 목표 Z %1 mm < Z 하한 %2 mm").arg(tgt.p[2], 0, 'f', 1).arg(linZMin_->value()));
            return;
        }
        const double reach = std::sqrt(tgt.p[0] * tgt.p[0] + tgt.p[1] * tgt.p[1] +
                                       (tgt.p[2] - LIN_SHOULDER_Z_MM) * (tgt.p[2] - LIN_SHOULDER_Z_MM));
        if (reach > LIN_MAX_REACH_MM) {
            log(QString("이동 거부: 목표가 작업 반경 밖입니다 (어깨에서 %1 mm > %2 mm)").arg(reach, 0, 'f', 0).arg(LIN_MAX_REACH_MM));
            return;
        }

        const pm::Vec6 tgtRpy = pm::xyzrpy_from_frame(tgt);
        if (confirmBig && (dist > LIN_CONFIRM_MM || ang > LIN_CONFIRM_DEG)) {
            const auto ans = QMessageBox::question(
                this, "MoveL 확인",
                QString("큰 이동입니다: 거리 %1 mm, 회전 %2°  (%3)\n\n"
                        "현재 [X,Y,Z | R,P,Y]: %4\n목표 [X,Y,Z | R,P,Y]: %5\n\n"
                        "경로에 사람/장애물이 없는지 확인하세요. 실행할까요?")
                    .arg(dist, 0, 'f', 1).arg(ang, 0, 'f', 1).arg(label)
                    .arg(vecStr(pm::xyzrpy_from_frame(cur))).arg(vecStr(tgtRpy)),
                QMessageBox::Yes | QMessageBox::No, QMessageBox::No);
            if (ans != QMessageBox::Yes) { log("MoveL 취소"); return; }
            // The dialog ran the event loop: the connection or robot state may have changed meanwhile.
            if (!robot_ || !canMove(&why)) { log("이동 불가: " + why); return; }
            Pose now;
            if (!read_tcp(now)) { log("이동 불가: TCP 피드백이 최신이 아닙니다"); return; }
            const pm::Frame n = pm::frame_from_posx(now);
            if (pm::dist(n.p, cur.p) > 0.5 || pm::rot_angle_deg(n.R, cur.R) > 0.2) {
                log("MoveL 취소: 확인하는 동안 로봇 자세가 바뀌었습니다. 다시 실행하세요");
                return;
            }
        }

        // Controller posx = ZYZ; at B = 0/180 keep the current A so the numbers stay close to the TP.
        const pm::Vec6 px = pm::posx_from_frame(tgt, tcp[3]);
        Pose posx;
        for (int i = 0; i < 6; ++i) posx[i] = static_cast<float>(px[i]);
        float vel[2] = {static_cast<float>(v), static_cast<float>(w)};
        float acc[2] = {static_cast<float>(a), static_cast<float>(aw)};
        g_fault = false;
        if (!robot_->amovel(posx.data(), vel, acc, 0.f, MOVE_MODE_ABSOLUTE, MOVE_REFERENCE_BASE)) {
            log("컨트롤러가 MoveL 명령을 거부했습니다");
            return;
        }
        moveKind_ = MoveKind::Linear;
        targetL_ = pm::frame_from_posx(posx);  // exactly what the controller received (float)
        totP_ = dist;
        totA_ = ang;
        maxSec_ = pm::trapezoid_time(dist, v, a) + pm::trapezoid_time(ang, w, aw) + 10.0;
        beginTracking(label);
        log(QString("이동 시작: %1 (%2 mm/s, %3°/s) 거리 %4 mm, 회전 %5° -> [X,Y,Z | R,P,Y] %6")
                .arg(label).arg(v).arg(w).arg(dist, 0, 'f', 1).arg(ang, 0, 'f', 1).arg(vecStr(tgtRpy)));
        log(QString("  amovel(posx %1, vel [%2, %3], acc [%4, %5], ABS, BASE)").arg(vecStr(px)).arg(v).arg(w).arg(a).arg(aw));
    }

    void beginTracking(const QString& label) {
        moveLabel_ = label;
        moveStart_ = Clock::now();
        idleSince_ = {};
        setPhase(Phase::Moving);
        mvShow("이동 중", C_RUN, 0, label);
    }

    // ---------- progress display ----------
    void mvShow(const QString& state, const char* color, double prog, const QString& info) {
        mvState_->setText(state);
        mvState_->setStyleSheet(QString("QLabel{background:%1;color:white;font-weight:bold;font-size:14px;padding:4px;}").arg(color));
        mvBar_->setValue(static_cast<int>(std::lround(std::clamp(prog, 0.0, 1.0) * 100)));
        mvInfo_->setText(info);
    }

    void finishMotion(const QString& state, const char* color, double prog, const QString& extra = "") {
        const double el = std::chrono::duration<double>(Clock::now() - moveStart_).count();
        mvShow(state, color, prog, QString("%1  (%2초)%3").arg(moveLabel_).arg(el, 0, 'f', 1).arg(extra));
        log(QString("[%1] %2  %3초%4").arg(state).arg(moveLabel_).arg(el, 0, 'f', 1).arg(extra));
        setPhase(Phase::Ready);
    }

    // Once per connection: the monitoring stream also carries the TCP rotation matrix. If it equals
    // Rz(A)*Ry(B)*Rz(C) of the reported posx, the ZYZ convention used for MoveL is confirmed on this
    // controller. Informational only (some firmware may leave the matrix empty).
    void checkRotationConvention() {
        Pose tcp;
        pm::Mat3 M{};
        double norm = 0;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            tcp = g_tcp;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c) { M[r][c] = g_rotm[r][c]; norm += M[r][c] * M[r][c]; }
        }
        if (!(norm > 0.5)) return;  // not provided
        const pm::Mat3 R = pm::zyz_to_R(tcp[3], tcp[4], tcp[5]), Mt = pm::transpose(M);
        double e = 0, et = 0;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c) {
                e = std::max(e, std::fabs(R[r][c] - M[r][c]));
                et = std::max(et, std::fabs(R[r][c] - Mt[r][c]));
            }
        if (std::min(e, et) < 1e-3)
            log("TCP 자세 규약 확인: 모니터링 회전행렬 = ZYZ(A,B,C) (MoveL RPY 변환 정상)");
        else
            log(QString("[주의] 모니터링 회전행렬이 ZYZ(A,B,C) 계산과 다릅니다 (오차 %1). "
                        "MoveL 회전은 작은 값으로 먼저 확인하세요").arg(std::min(e, et), 0, 'f', 4));
    }

    void log(const QString& s) {
        log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss.zzz ") + s);
    }

    static QString poseStr(const Pose& p, int prec = 2) {
        QStringList l;
        for (float v : p) l << QString::number(v, 'f', prec);
        return "[" + l.join(", ") + "]";
    }

    // Returns false (and leaves p untouched) when joint feedback cannot be trusted.
    bool readJoints(Pose& p) {
        if (!robot_ || !monitoring_fresh()) return false;
        LPROBOT_POSE j = robot_->get_current_posj();
        if (!j) return false;
        Pose r{};
        for (int i = 0; i < 6; ++i) {
            r[i] = j->_fPosition[i];
            if (!std::isfinite(r[i])) return false;
        }
        p = r;
        return true;
    }

    void applySpeed() {
        if (!robot_ || !g_access) { log("속도 적용 불가: 연결/제어권이 없습니다"); return; }
        if (phase_ == Phase::Moving) { log("속도 적용 불가: 이동 중입니다"); return; }
        const int v = speed_->value();
        const bool ok = robot_->change_operation_speed(static_cast<float>(v));
        log(QString("전체 속도 %1% 적용%2").arg(v).arg(ok ? "" : " 실패"));
    }

    void showHelp() {
        QMessageBox::information(
            this, "도움말 - A0509 제어 GUI",
            "<b>순서</b>: [연결] → TP 팝업 승인 (또는 '강제 제어권 요청') → 자동 Servo ON → 이동<br><br>"
            "<b>정지</b>: 빨간 정지 버튼 / Esc / Space (입력칸에 커서가 있어도 동작). 위급 시 TP 비상정지<br><br>"
            "<b>MoveJ</b>: 관절각 절대/상대. 30°보다 큰 변화는 확인 창<br>"
            "<b>MoveL</b>: X/Y/Z [mm] + Roll/Pitch/Yaw [°], 절대 / 상대(base) / 상대(tool). "
            "[현재값 불러오기] 후 일부만 고쳐 실행 권장. 100 mm 또는 30°보다 큰 이동은 확인 창<br>"
            "<b>조그</b>: 버튼 1회 = 1스텝. 직교(Base/Tool 기준)와 관절(J1~J6)<br><br>"
            "<b>안전장치</b>: Z 하한(기본 200 mm)보다 낮은 목표, 어깨에서 850 mm보다 먼 목표, 관절 한계를 넘는 목표는 거부. "
            "피드백 끊김·제어권 상실·비상정지·워치독 시간 초과 시 자동 정지<br><br>"
            "<b>좌표</b>: 이 로봇은 TCP 오프셋이 없어 X/Y/Z는 플랜지 면 기준입니다 (그리퍼 끝은 약 150 mm 아래). "
            "TP는 자세를 ZYZ(A,B,C)로 표시하며, GUI는 같은 자세를 RPY로 보여 줍니다<br><br>"
            "<b>전체 속도</b>: change_operation_speed. 컨트롤러에 유지되며 모든 이동 속도에 곱해집니다");
    }

    // ---------- connection sequence ----------
    void connectRobot() {
        g_access = g_denied = g_disconnected = g_fault = false;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_tcp_stamp = {};
        }
        robot_ = std::make_unique<CDRFLEx>();
        robot_->set_on_monitoring_access_control(on_access);
        robot_->set_on_monitoring_state(on_state);
        robot_->set_on_log_alarm(on_alarm);
        robot_->set_on_disconnected(on_disconnected);
        robot_->set_on_monitoring_data_ex(on_monitoring);

        const std::string ip = ip_->text().trimmed().toStdString();
        log(QString("연결 시도: %1").arg(ip.c_str()));
        if (!robot_->open_connection(ip, 12345)) {
            log("연결 실패");
            robot_.reset();
            refreshUi();
            return;
        }
        robot_->setup_monitoring_version(1);
        const auto monDeadline = Clock::now() + std::chrono::seconds(5);
        while (!monitoring_fresh() && Clock::now() < monDeadline) {
            QApplication::processEvents();
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!monitoring_fresh()) {
            log("모니터링 데이터 수신 실패 (관절값을 신뢰할 수 없음). 연결 해제");
            disconnectRobot();
            return;
        }
        SYSTEM_VERSION v{};
        robot_->get_system_version(&v);
        log(QString("연결됨: model=%1 controller=%2 SDK=%3")
                .arg(v._szRobotModel).arg(v._szController).arg(robot_->get_library_version()));
        if (robot_->get_robot_system() != ROBOT_SYSTEM_REAL) {
            log("컨트롤러가 REAL 시스템이 아닙니다. 연결 해제");
            disconnectRobot();
            return;
        }
        checkRotationConvention();
        modeChanged_ = false;
        haveHome_ = false;
        const bool force = force_->isChecked();
        robot_->manage_access_control(force ? MANAGE_ACCESS_CONTROL_FORCE_REQUEST : MANAGE_ACCESS_CONTROL_REQUEST);
        log(force ? "제어권 강제 요청" : "제어권 요청 - TP 팝업에서 승인하세요");
        setPhase(Phase::WaitAccess, 15);
    }

    void beginServo() {
        if (!robot_ || !g_access) return;
        const ROBOT_STATE s = robot_->get_robot_state();
        if (s == STATE_STANDBY) { enterReady(); return; }
        if (s != STATE_SAFE_OFF && s != STATE_SAFE_OFF2) {
            log(QString("Servo ON 거부: 상태 %1. TP에서 비상정지/보호정지를 먼저 해제하세요").arg(state_name(s)));
            setPhase(Phase::Ready);
            return;
        }
        g_fault = false;
        log("Servo ON 요청");
        if (!robot_->set_robot_control(CONTROL_SERVO_ON)) {
            log("Servo ON 요청이 거부되었습니다");
            setPhase(Phase::Ready);
            return;
        }
        setPhase(Phase::WaitServo, 10);
    }

    void enterReady() {
        if (!modeChanged_) {
            prevMode_ = robot_->get_robot_mode();
            modeChanged_ = true;
        }
        if (robot_->get_robot_mode() != ROBOT_MODE_AUTONOMOUS && !robot_->set_robot_mode(ROBOT_MODE_AUTONOMOUS)) {
            log("AUTONOMOUS 모드 전환 실패");
        }
        if (!haveHome_) {
            Pose p;
            if (readJoints(p)) {
                home_ = p;
                haveHome_ = true;
                log("원위치 저장: " + poseStr(home_));
            } else {
                log("원위치 자동 저장 실패: 관절 피드백이 최신이 아닙니다. [원위치 저장]을 눌러주세요");
            }
        }
        log("준비 완료 (STANDBY, AUTONOMOUS)");
        setPhase(Phase::Ready);
    }

    void disconnectRobot() {
        if (!robot_) return;
        if (!g_disconnected) {
            if (phase_ == Phase::Moving) robot_->stop(STOP_TYPE_QUICK);
            if (modeChanged_ && robot_->get_robot_mode() != prevMode_) {
                const bool ok = robot_->set_robot_mode(prevMode_);
                log(QString("로봇 모드 복원 -> %1%2").arg(prevMode_).arg(ok ? "" : " (실패)"));
            }
            robot_->close_connection();
        }
        if (phase_ == Phase::Moving) mvShow("✖ 중단됨", C_FAIL, mvBar_->value() / 100.0, moveLabel_ + "  - 연결 해제");
        robot_.reset();
        log("연결 해제");
        setPhase(Phase::Off);
    }

    void setPhase(Phase p, int timeoutSec = 0) {
        phase_ = p;
        deadline_ = Clock::now() + std::chrono::seconds(timeoutSec);
        refreshUi();
    }

    bool canMove(QString* why) {
        if (phase_ != Phase::Ready) { *why = "준비 상태가 아닙니다"; return false; }
        if (!g_access) { *why = "제어권이 없습니다"; return false; }
        if (robot_->get_robot_state() != STATE_STANDBY) { *why = "STANDBY 상태가 아닙니다"; return false; }
        if (robot_->check_motion() != 0) { *why = "로봇이 이미 움직이는 중입니다"; return false; }
        if (robot_->get_robot_mode() != ROBOT_MODE_AUTONOMOUS) { *why = "AUTONOMOUS 모드가 아닙니다"; return false; }
        return true;
    }

    void stopNow(const QString& reason, bool fault = false) {
        if (!robot_) return;
        robot_->stop(STOP_TYPE_QUICK);
        log("[정지] " + reason + " -> stop(QUICK)");
        if (phase_ == Phase::Moving)
            finishMotion(fault ? "✖ 안전정지" : "■ 정지됨", fault ? C_FAIL : C_STOP, mvBar_->value() / 100.0, "  - " + reason);
    }

    // cur = joints [deg], tcp = TCP [mm, ZYZ deg]; both fresh (checked by tick()).
    void pollMotion(ROBOT_STATE st, const Pose& cur, const Pose& tcp) {
        const double el = std::chrono::duration<double>(Clock::now() - moveStart_).count();
        if (g_fault || is_fault(st)) { stopNow(QString("로봇 이상 상태 %1").arg(state_name(st)), true); return; }
        if (!g_access) { stopNow("제어권 상실", true); return; }
        if (el > maxSec_) { stopNow(QString("워치독 시간 초과 (%1s)").arg(maxSec_, 0, 'f', 1), true); return; }

        bool reached;
        double prog;
        QString rem;
        if (moveKind_ == MoveKind::Joint) {
            double r = 0;
            for (int i = 0; i < 6; ++i) r = std::max(r, double(std::fabs(cur[i] - targetJ_[i])));
            reached = r <= JOINT_TOL_DEG;
            prog = totJ_ > 1e-4 ? 1.0 - r / totJ_ : 1.0;
            rem = QString("남은 각도 %1°").arg(r, 0, 'f', 1);
        } else {
            const pm::Frame c = pm::frame_from_posx(tcp);
            const double pe = pm::dist(c.p, targetL_.p), re = pm::rot_angle_deg(c.R, targetL_.R);
            reached = pe <= LIN_TOL_MM && re <= LIN_TOL_DEG;
            const double fp = totP_ > 0.01 ? pe / totP_ : 0.0, fa = totA_ > 0.01 ? re / totA_ : 0.0;
            prog = 1.0 - std::max(fp, fa);
            rem = QString("남은 거리 %1 mm, 회전 %2°").arg(pe, 0, 'f', 1).arg(re, 0, 'f', 1);
        }
        mvShow("이동 중", C_RUN, prog, QString("%1  |  %2  |  %3초").arg(moveLabel_).arg(rem).arg(el, 0, 'f', 1));
        if (el < 0.2) return;

        if (robot_->check_motion() == 0 && st == STATE_STANDBY) {
            if (reached) { finishMotion("✔ 이동 완료", C_OK, 1.0); return; }
            if (idleSince_ == Clock::time_point{}) idleSince_ = Clock::now();
            if (Clock::now() - idleSince_ > std::chrono::milliseconds(1500))
                finishMotion("✖ 중단됨", C_FAIL, prog, "  - 목표 도달 전에 멈춤 (" + rem + ")");
        } else {
            idleSince_ = {};
        }
    }

    // ---------- periodic update ----------
    void tick() {
        std::vector<std::string> logs;
        Pose tcp;
        bool tcpFresh;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            logs.swap(g_logs);
            tcp = g_tcp;
            tcpFresh = fresh_locked();
            for (float v : tcp) tcpFresh = tcpFresh && std::isfinite(v);
        }
        for (auto& s : logs) log(QString::fromStdString(s));
        if (!robot_) return;
        if (g_disconnected) { disconnectRobot(); return; }

        const ROBOT_STATE st = robot_->get_robot_state();
        const ROBOT_MODE mode = robot_->get_robot_mode();
        Pose cur{};
        const bool jointsFresh = readJoints(cur);
        for (int j = 0; j < 6; ++j) curJ_[j]->setText(jointsFresh ? QString::number(cur[j], 'f', 2) : QString("-"));
        if (phase_ == Phase::Moving && (!jointsFresh || !tcpFresh)) {
            stopNow(jointsFresh ? "TCP 피드백 끊김" : "관절 피드백 끊김", true);
            refreshButtons(st);
            return;
        }
        if (tcpFresh) {
            const pm::Vec6 r = pm::xyzrpy_from_frame(pm::frame_from_posx(tcp));
            for (int i = 0; i < 6; ++i) curP_[i]->setText(QString::number(r[i], 'f', 2));
            zyzLbl_->setText(QString("TP 표시 (ZYZ):  A %1   B %2   C %3 [°]")
                                 .arg(tcp[3], 0, 'f', 2).arg(tcp[4], 0, 'f', 2).arg(tcp[5], 0, 'f', 2));
        } else {
            for (auto* l : curP_) l->setText("-");
            zyzLbl_->setText("TP 표시 (ZYZ): (모니터링 데이터 없음)");
        }
        setBadge(stateLbl_, state_name(st), is_fault(st) ? C_FAIL : st == STATE_STANDBY ? C_OK : st == STATE_MOVING ? C_RUN : C_IDLE);
        setBadge(modeLbl_, mode == ROBOT_MODE_AUTONOMOUS ? "AUTONOMOUS" : mode == ROBOT_MODE_MANUAL ? "MANUAL" : "기타",
                 mode == ROBOT_MODE_AUTONOMOUS ? C_OK : C_IDLE);
        setBadge(accessLbl_, g_access ? "있음" : "없음", g_access ? C_OK : C_FAIL);

        switch (phase_) {
        case Phase::WaitAccess:
            if (g_access) beginServo();
            else if (g_denied || Clock::now() > deadline_) {
                log(g_denied ? "제어권 요청이 거부되었습니다" : "제어권 대기 시간 초과");
                disconnectRobot();
            }
            break;
        case Phase::WaitServo:
            if (st == STATE_STANDBY) enterReady();
            else if (Clock::now() > deadline_) {
                log(QString("Servo ON 시간 초과 (상태 %1)").arg(state_name(st)));
                setPhase(Phase::Ready);
            }
            break;
        case Phase::Moving:
            pollMotion(st, cur, tcp);
            break;
        default:
            break;
        }
        refreshButtons(st);
    }

    void refreshUi() {
        const bool on = robot_ != nullptr;
        connectBtn_->setText(on ? "연결 해제" : "연결");
        ip_->setEnabled(!on);
        force_->setEnabled(!on);
        setBadge(connLbl_, on ? "ON" : "OFF", on ? C_OK : C_IDLE);
        if (!on) {
            for (auto* l : {stateLbl_, modeLbl_, accessLbl_}) setBadge(l, "-", nullptr);
            zyzLbl_->setText("TP 표시 (ZYZ): -");
            for (auto* l : curJ_) l->setText("-");
            for (auto* l : curP_) l->setText("-");
        }
        refreshButtons(on ? robot_->get_robot_state() : STATE_NOT_READY);
    }

    void refreshButtons(ROBOT_STATE st) {
        const bool fresh = robot_ && monitoring_fresh();
        const bool ready = fresh && phase_ == Phase::Ready && g_access && st == STATE_STANDBY;
        for (int i = 0; i < 6; ++i)
            for (auto* b : {cPlus_[i], cMinus_[i], jPlus_[i], jMinus_[i]}) b->setEnabled(ready);
        for (auto* b : {jGoBtn_, linGoBtn_, setHomeBtn_}) b->setEnabled(ready);
        homeBtn_->setEnabled(ready && haveHome_);
        jHomeValBtn_->setEnabled(haveHome_);
        jLoadBtn_->setEnabled(fresh);
        linLoadBtn_->setEnabled(fresh);
        speedBtn_->setEnabled(robot_ && g_access && phase_ != Phase::Moving);
        servoBtn_->setEnabled(robot_ && phase_ == Phase::Ready && g_access &&
                              (st == STATE_SAFE_OFF || st == STATE_SAFE_OFF2));
        stopBtn_->setEnabled(robot_ != nullptr);
        connectBtn_->setEnabled(phase_ != Phase::Moving);
    }

    // widgets
    QLineEdit* ip_;
    QCheckBox* force_;
    QPushButton *connectBtn_, *servoBtn_, *stopBtn_, *speedBtn_, *helpBtn_;
    QLabel *connLbl_, *stateLbl_, *modeLbl_, *accessLbl_, *zyzLbl_;
    QLabel *curJ_[6], *curP_[6];
    // MoveJ panel
    QDoubleSpinBox* jIn_[6] = {};
    QDoubleSpinBox *jVel_, *jAcc_;
    QRadioButton *jAbs_, *jRel_;
    QPushButton *jLoadBtn_, *jHomeValBtn_, *homeBtn_, *setHomeBtn_, *jGoBtn_;
    // MoveL panel
    QDoubleSpinBox* lin_[6] = {};  // X, Y, Z [mm], Roll, Pitch, Yaw [deg]
    QDoubleSpinBox *linVel_, *linAcc_, *linRotVel_, *linRotAcc_;
    QDoubleSpinBox* linZMin_ = nullptr;  // Z floor [mm] for MoveL and Cartesian jog
    QRadioButton *linAbs_, *linRelBase_, *linRelTool_;
    QPushButton *linLoadBtn_, *linZeroBtn_, *linGoBtn_;
    // jog panel
    QRadioButton *jogBase_, *jogTool_;
    QComboBox *jogMm_, *jogDeg_, *jogJDeg_;
    QDoubleSpinBox *jogV_, *jogW_, *jogJV_;
    QPushButton *cPlus_[6], *cMinus_[6], *jPlus_[6], *jMinus_[6];
    // progress + bottom
    QLabel *mvState_, *mvInfo_;
    QProgressBar* mvBar_;
    QSlider* speed_;
    QLabel* speedLbl_;
    QPlainTextEdit* log_;
    QTimer timer_;

    // robot
    std::unique_ptr<CDRFLEx> robot_;
    Phase phase_ = Phase::Off;
    MoveKind moveKind_ = MoveKind::Joint;
    Clock::time_point deadline_, moveStart_, idleSince_;
    Pose home_{}, targetJ_{};
    pm::Frame targetL_{};
    double totJ_ = 0, totP_ = 0, totA_ = 0;
    QString moveLabel_;
    bool haveHome_ = false, modeChanged_ = false;
    ROBOT_MODE prevMode_ = ROBOT_MODE_MANUAL;
    double maxSec_ = 0;
};

int main(int argc, char** argv) {
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}
