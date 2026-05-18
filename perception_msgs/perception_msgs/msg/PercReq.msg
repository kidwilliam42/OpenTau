
# 动作执行id,增加含义：时间+序号 
uint64 req_id

# Perceive Kind

# 自动充电
uint32 PERC_AUTO_CHARGING        = 20
# 环境查看
uint32 PERC_CHECK_ENV            = 21
# 自动建图
uint32 PERC_AUTO_MAPPING         = 22

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

uint32 PERC_HAND_OK              = 70
uint32 PERC_HAND_V               = 71
uint32 PERC_HAND_SHAKE           = 72
uint32 PERC_HAND_COME            = 73
uint32 PERC_HAND_WAVE            = 74
uint32 PERC_CAMERA_DEMO          = 80

# 任务取消（不需要请求）
uint32 PERC_CANCEL               = 100

# 感知类型
uint32 perc_kind

# 危险信息
string danger_info