std_msgs/Header header

# ActionState
uint8 ACTION_IDLE = 0
uint8 ACTION_RUNNING = 1
uint8 ACTION_DONE = 2

# ActionResult
uint8 ACTION_NONE = 0
uint8 ACTION_SUCCESS = 1
uint8 ACTION_FAIL = 2

# ErrCode
uint16 ACTION_ERR_NONE = 0
uint16 ACTION_ERR_NOWORK = 1

# 当前执行的id
uint64 action_id


# Perceive Kind
uint32 PERC_DEST                 = 1
uint32 PERC_ABS                  = 2
uint32 PERC_REL                  = 3

# 自动充电
uint32 PERC_AUTO_CHARGING        = 20
# 环境查看
uint32 PERC_CHECK_ENV            = 21
# 自动建图
uint32 PERC_AUTO_MAPPING         = 22
# 抓取任务
uint32 PERC_AUTO_GRIP            = 23

# 跟随模式
uint32 PERC_FOLLOW               = 30
# 陌生人迎接
uint32 PERC_STRANGER_WELCOME     = 31
# 主人迎接
uint32 PERC_OWNER_WELCOME        = 32
# 行为互动
uint32 PERC_BEHAVIOR_INTERACTION = 33
# 握手动作
uint32 PERC_HANDSHAKE            = 34

# 待机陪伴
uint32 PERC_STANDBY_ACCOMPANY    = 35
# 娱乐陪伴
uint32 PERC_HAPPY_ACCOMPANY      = 36
# 工作陪伴
uint32 PERC_WORK_ACCOMPANY       = 37
# 积极陪伴
uint32 PERC_POSITIVE_ACCOMPANY   = 38
# 消极陪伴
uint32 PERC_NEGATIVE_ACCOMPANY   = 39

# 小孩看护
uint32 PERC_CHILD_WATCH          = 40
# 老人看护
uint32 PERC_OLDER_WATCH          = 41
# 危险提醒
uint32 PERC_DANGER_REMINDER      = 42
# 事件提醒
uint32 PERC_EVENT_REMINDER       = 43
# 开始走秀（语言下方，不需要感知）
uint32 PERC_CATWALK_START        = 44
# 走秀互动
uint32 PERC_CATWALK_ACT          = 45
# 走秀展示
uint32 PERC_CATWALK_SHOW         = 46
# 走秀返回
uint32 PERC_CATWALK_BACK         = 47

# 自由活动
uint32 PERC_FREE_TIME            = 50
# 社交模式
uint32 PERC_SOCIALIZE            = 51
# 围观场景
uint32 PERC_SPECT_PERFORM        = 55
# 表情分析
uint32 PERC_FACE_DETCTION        = 56


# 上报危险信息（不需要请求）
uint32 PERC_DANGER_INFO          = 60
# 欢迎模式
uint32 PERC_WELCOME_DEMO         = 61
# 回大厅找人
uint32 PERC_LOBBY_DEMO           = 62

# 任务取消（不需要请求）
uint32 PERC_CANCEL               = 100

uint32 PERC_NODE_CLOSE           = 200
uint32 PERC_NODE_START           = 201
uint32 PERC_NODE_RESET           = 202

# 感知类型
uint32 perc_kind
# 执行状态：空闲/正在执行/执行完成
uint8 exe_state
# 执行结果：无效/成功/失败
uint8 exe_result

# 错误码
uint16 err_code
# 错误消息
string err_info
# 剩余执行时间
uint64 wait_for_millisec
