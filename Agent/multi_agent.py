# %%
# ==================== 第一部分：导入必要的库 ====================
import os  # 操作系统接口，用于读取环境变量、文件路径操作
import re  # 正则表达式，用于解析文本中的JSON和数字
import json  # 处理JSON数据，模拟工具返回JSON格式结果
import logging  # 日志模块，替代print输出，记录程序运行信息
from typing import List, Dict, Any, TypedDict, Literal  # 类型注解，提高代码可读性和可维护性
from datetime import datetime  # 获取当前时间戳，用于记录消息处理时间

import dotenv  # 从 .env 文件加载环境变量（如API密钥）
from langchain_openai import ChatOpenAI  # OpenAI 兼容的聊天模型接口，此处对接DeepSeek
from langchain_core.messages import HumanMessage, AIMessage  # 人类和AI消息类型，用于构建对话历史
from langchain_core.prompts import ChatPromptTemplate  # 提示模板，构建 prompts
from langchain_core.output_parsers import StrOutputParser  # 字符串输出解析器，提取模型返回的纯文本
from langchain_core.tools import tool  # 将函数转换为LangChain工具，供Agent调用
from langgraph.graph import StateGraph, START, END  # 状态图核心类，构建有状态工作流
from langchain.agents import create_agent  # 快速创建Agent（模型+工具+系统提示）

# %%
# ==================== 第二部分：配置日志与环境变量 ====================
# 配置日志：级别为INFO，格式包含时间、模块名、日志级别和消息
logging.basicConfig(
    level=logging.INFO,                                # 设置日志级别为INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # 日志输出格式
)
logger = logging.getLogger(__name__)                   # 获取当前模块的日志器，用于记录本文件内的日志

# 加载 .env 文件中的环境变量
dotenv.load_dotenv()                                   # 从项目根目录的 .env 文件加载环境变量

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")       # 读取DeepSeek的API密钥
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")     # 读取DeepSeek的API基础URL

_MODEL_CACHE = None                                    # 模型实例全局缓存，避免重复初始化

# 验证必要的环境变量是否存在，缺失则抛出错误并终止程序
if not DEEPSEEK_API_KEY:
    raise ValueError("环境变量 DEEPSEEK_API_KEY 未设置，请在 .env 文件中添加该变量。")
if not DEEPSEEK_BASE_URL:
    raise ValueError("环境变量 DEEPSEEK_BASE_URL 未设置，请在 .env 文件中添加该变量。")


def get_model():
    """
    获取聊天模型的单例实例（DeepSeek Chat）。
    使用全局缓存，避免重复创建，节省资源。
    """
    global _MODEL_CACHE                                # 声明要修改全局变量
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE                            # 缓存存在则直接返回
    # 创建 ChatOpenAI 实例，配置模型名、温度、最大token、API密钥和基础URL
    _MODEL_CACHE = ChatOpenAI(
        model="deepseek-chat",                         # 使用的模型名称
        temperature=0.2,                               # 较低温度使生成更稳定、确定
        max_tokens=1000,                               # 限制单次生成的最大token数
        api_key=DEEPSEEK_API_KEY,                      # 使用从环境变量读取的API密钥
        base_url=DEEPSEEK_BASE_URL                     # API基础URL
    )
    return _MODEL_CACHE


# %%
# ==================== 第三部分：模拟数据 ====================
# 模拟订单数据库，键为订单号，值为订单详细信息
MOCK_ORDERS = {
    "ORD001": {                                        # 订单001的信息
        "status": "已发货",                            # 订单状态
        "product": "智能手表 Pro",                     # 商品名称
        "price": 1299,                                 # 价格
        "shipping": "顺丰快递",                        # 快递公司
        "tracking": "SF1234567890",                    # 快递单号
        "estimated_delivery": "2024-12-20"             # 预计送达日期
    },
    "ORD002": {                                        # 订单002的信息
        "status": "处理中",
        "product": "无线耳机 Max",
        "price": 899,
        "shipping": "待发货",
        "tracking": None,                              # 暂无快递单号
        "estimated_delivery": "2024-12-22"
    },
    "ORD003": {                                        # 订单003的信息
        "status": "已完成",
        "product": "便携充电宝",
        "price": 199,
        "shipping": "已签收",
        "tracking": "YT9876543210",
        "estimated_delivery": "2024-12-15"
    }
}

# 模拟产品数据库，键为产品名称，值为价格、特性、库存和评分
MOCK_PRODUCTS = {
    "智能手表 Pro": {                                  # 产品“智能手表 Pro”的信息
        "price": 1299,                                 # 价格
        "features": ["心率监测", "GPS定位", "防水50米", "7天续航"],  # 产品特性列表
        "stock": 50,                                   # 库存数量
        "rating": 4.8                                  # 用户评分
    },
    "无线耳机 Max": {
        "price": 899,
        "features": ["主动降噪", "40小时续航", "蓝牙5.3", "通话降噪"],
        "stock": 120,
        "rating": 4.6
    },
    "便携充电宝": {
        "price": 199,
        "features": ["20000mAh", "快充支持", "双USB输出", "LED显示"],
        "stock": 200,
        "rating": 4.5
    },
    "智能音箱": {
        "price": 499,
        "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi音质"],
        "stock": 80,
        "rating": 4.7
    }
}

# 常见问题数据库，键为问题类型，值为标准答案
FAQ_DATABASE = {
    "连接问题": "请尝试以下步骤：1) 重启设备 2) 检查蓝牙是否开启 3) 删除配对记录后重新配对 4) 确保设备电量充足",
    "充电问题": "建议使用原装充电器，检查充电线是否损坏。如果问题持续，可能需要更换电池或送修。",
    "软件更新": "打开设备对应的APP，进入设置-关于-检查更新，按提示操作即可完成更新。",
    "退货政策": "我们支持7天无理由退货，30天内有质量问题可换货。请保留好购买凭证和完整包装。"
}


# %%
# ==================== 第四部分：工具函数（Agent可调用的“技能”） ====================
@tool                                                          # 将函数标记为 LangChain 工具，Agent 可以调用
def query_order(order_id: str) -> str:
    """查询订单信息，根据订单号返回订单详情JSON字符串"""
    order = MOCK_ORDERS.get(order_id.upper())                  # 统一转大写后查找，确保不区分大小写
    if order:
        # 找到订单，以格式化的JSON返回
        return json.dumps(order, ensure_ascii=False, indent=2)
    return f"未找到订单{order_id}"                              # 未找到时返回提示信息


@tool
def track_shipping(tracking_number: str) -> str:
    """查询物流信息，根据快递单号返回物流状态描述"""
    if tracking_number.startswith("SF"):                       # 顺丰快递单号以SF开头
        return f"顺丰快递{tracking_number}:包裹已到达配送站，预计今日送达"
    elif tracking_number.startswith("YT"):                     # 圆通快递单号以YT开头
        return f"圆通快递{tracking_number}:已签收"
    return f"未找到物流信息{tracking_number}"                   # 未匹配到快递公司


@tool
def search_product(keyword: str) -> str:
    """搜索产品信息，根据关键词在产品名称中匹配，返回匹配产品列表JSON"""
    results = []                                               # 存储匹配结果
    for name, info in MOCK_PRODUCTS.items():
        if keyword.lower() in name.lower():                    # 大小写不敏感匹配
            results.append({
                "name": name,
                "price": info["price"],
                "features": info["features"],
                "rating": info["rating"]
            })
    if results:
        return json.dumps(results, ensure_ascii=False, indent=2)
    return f"未找到包含{keyword}的产品"


@tool
def get_product_recommendations(budget: int) -> str:
    """根据预算推荐产品，返回价格不超过预算且评分最高的前3款产品JSON"""
    recommendations = []                                       # 存储推荐列表
    for name, info in MOCK_PRODUCTS.items():
        if info['price'] <= budget:                            # 价格符合预算才加入
            recommendations.append({
                "name": name,
                "price": info['price'],
                "rating": info['rating']
            })
    # 按价格从高到低排序（更贵的产品通常功能更强）
    recommendations.sort(key=lambda x: x["price"], reverse=True)
    if recommendations:
        return json.dumps(recommendations[:3], ensure_ascii=False, indent=2)  # 返回前3个
    return f"在预算{budget}内暂无推荐产品"


@tool
def search_faq(problem_type: str) -> str:
    """搜索常见问题解答，根据问题类型关键词匹配FAQ答案"""
    for key, answer in FAQ_DATABASE.items():
        # 双向包含匹配（如“连接”能匹配到“连接问题”，“充电问题”也能匹配到“充电”）
        if problem_type in key or key in problem_type:
            return f"【{key}】\n{answer}"
    return "未找到相关FAQ，建议联系人工客服获取更多帮助。"


# %%
# ==================== 第五部分：客服系统状态定义 ====================
class CustomerServiceState(TypedDict):
    """定义智能客服系统在工作流中使用的状态字段，所有节点共享并更新此状态"""
    user_message: str                      # 用户输入的原始消息
    chat_history: List[Dict[str, str]]     # 对话历史，格式 [{"role": "user", "content": "..."}]
    intent: str                            # 识别出的用户意图
    confidence: float                      # 意图分类的置信度 (0-1)
    agent_response: str                    # 最终客服回复内容
    needs_escalation: bool                 # 是否需要升级到人工客服
    escalation_reason: str                 # 升级原因描述
    quality_score: float                   # 质检评分 (0-1)
    already_escalated: bool                # 本轮是否已执行过升级（避免重复添加升级提示）
    metadata: Dict[str, Any]               # 附加元数据（如时间戳等）


# %%
# ==================== 第六部分：安全JSON解析工具 ====================
def safe_parse_json(text: str, default: dict = None) -> dict:
    """
    安全解析 LLM 返回的可能包含在 Markdown 代码块中的 JSON 字符串。
    如果解析失败，返回默认值，避免程序因 JSON 格式错误而崩溃。
    """
    if default is None:
        default = {}                       # 默认空字典

    content = text.strip()                 # 去除首尾空白

    # 策略1：优先提取 Markdown JSON 代码块（```json ... ```）
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
    if json_match:
        content = json_match.group(1).strip()  # 提取代码块中的内容
    else:
        # 策略2：尝试提取第一个花括号包裹的 JSON 对象（非贪婪匹配，可能无法处理嵌套）
        brace_match = re.search(r'\{.*?\}', content)
        if brace_match:
            content = brace_match.group(0)     # 提取花括号内容

    try:
        return json.loads(content)             # 尝试解析JSON
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return default                         # 返回安全的默认值，避免程序中断


# %%
# ==================== 第七部分：意图分类器 ====================
class IntentClassifier:
    """使用 LLM 对用户消息进行意图分类，返回意图类型和置信度"""
    VALID_INTENTS = {"tech_support", "order_service", "product_consult", "general_chat", "escalate"}  # 合法的意图集合

    def __init__(self):
        self.llm = get_model()                 # 获取模型实例
        # 定义意图分类提示模板：要求 LLM 以 JSON 格式返回意图和置信度
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个意图分类专家，分析用户消息并返回意图分类。

可选意图：
- tech_support: 具体技术问题、故障排除、使用帮助（如“蓝牙连不上”、“充电慢”）
- order_service: 具体订单查询、物流跟踪、退换货（如“查订单 ORD001”、“快递到哪了”）
- product_consult: 具体产品咨询、价格询问、功能介绍（如“智能手表多少钱”、“推荐一款耳机”）
- general_chat: 通用对话、闲聊、功能询问、模糊问题、非业务问题（如“你好”、“你能做什么”、“帮我写首诗”、“我不懂”）
- escalate: 明确要求人工客服、投诉、严重不满、要求经理、连续无法解决问题（如“我要投诉”、“转人工”、“叫你们经理来”）

返回格式(JSON):
{{"intent": "意图类型","confidence": 0.0-1.0, "reason": "分类原因"}}

只返回JSON，不要其他内容。"""),
            ("human", "{message}")
        ])

    def classify(self, message: str) -> Dict[str, Any]:
        """分析用户消息，返回包含意图、置信度和原因的字典"""
        chain = self.prompt | self.llm | StrOutputParser()    # 构建链：提示 -> 模型 -> 字符串输出
        result = chain.invoke({"message": message})           # 调用模型获取分类结果

        # 设置默认结果（解析失败时使用）
        default_result = {"intent": "general_chat", "confidence": 0.5, "reason": "解析失败"}
        parsed = safe_parse_json(result, default_result)      # 安全解析JSON

        # 校验意图的合法性：如果不在预定义集合中，则标记为 general_chat
        intent = parsed.get("intent", "general_chat")
        if intent not in self.VALID_INTENTS:
            intent = "general_chat"
        parsed["intent"] = intent

        return parsed


# %%
# ==================== 第八部分：专业Agent基类 ====================
class BaseAgent:
    """Agent基类，提供将对话历史转换为 LangChain 消息对象的通用方法"""

    @staticmethod
    def _prepare_messages(message: str, chat_history: List[Dict] = None, max_history: int = 6):
        """将存储为字典的对话历史转换为 HumanMessage/AIMessage 列表，并添加当前用户消息"""
        if chat_history is None:
            chat_history = []                                  # 防止 None 切片报错
        messages = []                                          # 初始化消息列表
        for msg in chat_history[-max_history:]:                # 只取最近 max_history 条，避免 token 超限
            role = msg["role"]
            if role in ("user", "human"):                      # 用户消息
                messages.append(HumanMessage(content=msg["content"]))
            elif role in ("assistant", "ai"):                  # 助手消息
                messages.append(AIMessage(content=msg["content"]))
            else:                                              # 未知角色默认按用户消息处理
                messages.append(HumanMessage(content=msg["content"]))
        messages.append(HumanMessage(content=message))         # 追加当前用户消息
        return messages


# %%
# ==================== 第九部分：各专业领域Agent ====================
class TechSupportAgent(BaseAgent):
    """技术支持Agent，负责技术问题解答，拥有搜索FAQ的工具"""

    def __init__(self):
        super().__init__()                                     # 调用基类初始化
        self.llm = get_model()                                 # 获取模型
        self.tools = [search_faq]                              # 可用工具列表：仅搜索FAQ
        self.system_prompt = """你是一个专业的技术支持工程师，你的职责是：
1.分析用户遇到的技术问题
2.提供清晰的故障排除步骤
3.使用search_faq工具查找相关解决方案
4.如果问题超出能力范围，建议升级到人工支持
回复要求：
- 语气友好专业
- 步骤清晰有序
- 提供多个可能解决方案"""
        # 使用 create_agent 创建一个能调用工具的Agent实例
        self.agent = create_agent(
            model=self.llm,                                    # 使用的语言模型
            tools=self.tools,                                  # 可调用的工具列表
            system_prompt=self.system_prompt,                  # 系统提示词，定义Agent角色
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        """处理用户消息，返回Agent的回复文本"""
        messages = self._prepare_messages(message, chat_history)  # 准备消息列表（含历史）
        result = self.agent.invoke({"messages": messages})        # 调用Agent
        if result.get("messages"):
            return result["messages"][-1].content                # 取最后一条消息（即Agent回复）
        return "抱歉，我暂时无法处理您的问题。建议联系人工客服"


class OrderServiceAgent(BaseAgent):
    """订单服务Agent，负责订单查询和物流跟踪，拥有查询订单和物流的工具"""

    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [query_order, track_shipping]             # 可用工具：查询订单、跟踪物流
        self.system_prompt = """你是一个专业的订单服务专员。你的职责是：
1. 帮助用户查询订单状态
2. 提供物流跟踪信息
3. 解答退换货相关问题
4. 使用工具获取准确信息

回复要求：
- 信息准确完整
- 主动提供相关信息
- 如果需要订单号，礼貌询问"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = self.agent.invoke({"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，订单查询服务暂时不可用，请稍后再试。"


class ProductConsultAgent(BaseAgent):
    """产品咨询Agent，负责产品推荐和功能介绍，拥有搜索产品和推荐产品的工具"""

    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = [search_product, get_product_recommendations]  # 可用工具：搜索产品、推荐产品
        self.system_prompt = """你是一个热情的产品顾问。你的职责是：
1. 介绍产品功能和特点
2. 根据用户需求推荐合适的产品
3. 解答价格和库存问题
4. 使用工具获取最新产品信息

回复要求：
- 热情有亲和力
- 突出产品优势
- 根据用户需求推荐
- 不要过度推销"""
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = self.agent.invoke({"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "抱歉，产品信息查询暂时不可用。请稍后再试。"


class GeneralChatAgent(BaseAgent):
    """通用对话Agent，像人一样回应各种非业务问题（闲聊、功能询问、模糊问题等）"""

    def __init__(self):
        super().__init__()
        self.llm = get_model()
        self.tools = []                                        # 不需要工具
        self.system_prompt = """你是一个友善、耐心、像人一样的智能客服助手。
你可以处理任何问题，包括：
- 闲聊：你好、今天天气、心情如何
- 功能询问：你能做什么、怎么使用
- 模糊问题：我不太清楚、怎么办
- 非业务问题：帮我写首诗、讲个笑话

回答要求：
- 像真正的客服人员一样自然、温和、体贴
- 当用户问你能做什么时，主动介绍自己的业务范围（订单查询、产品咨询、技术支持等）
- 如果实在无法回答，可以说“这个问题我暂时不太擅长，但你可以具体告诉我需要什么帮助吗？”
- 绝对不要直接建议转人工，除非用户明确要求或者问题涉及敏感内容

保持人性化的语气，不要像机器人一样死板。"""

        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
        )

    def handle(self, message: str, chat_history: List = None) -> str:
        messages = self._prepare_messages(message, chat_history)
        result = self.agent.invoke({"messages": messages})
        if result.get("messages"):
            return result["messages"][-1].content
        return "嗯...我还在学习中，你能再说一遍吗？"


# %%
# ==================== 第十部分：质量检查器 ====================
class QualityChecker:
    """使用 LLM 对客服回复进行评分，判断是否需要升级到人工"""

    def __init__(self):
        self.llm = get_model()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是客服质量检查专家。评估客服回复的质量。

评估维度：
1. 相关性（0-25分）：回复是否针对用户问题
2. 完整性（0-25分）：是否提供了足够的信息
3. 专业性（0-25分）：语言是否专业得体
4. 有用性（0-25分）：是否真正帮助到用户

返回格式（JSON）：
{{"total_score": 0-100, "needs_escalation": True/False, "reason": "评估说明"}}

只返回JSON。"""),
            ("human", """用户问题：{user_message}
客服回复：{agent_response}

请评估：""")
        ])

    def check(self, user_message: str, agent_response: str) -> Dict[str, Any]:
        """对回复进行质量检查，返回包含总分和升级建议的字典"""
        chain = self.prompt | self.llm | StrOutputParser()    # 构建链
        result = chain.invoke({
            "user_message": user_message,
            "agent_response": agent_response
        })
        # 安全解析 JSON，失败时使用默认值
        default_result = {"total_score": 60, "needs_escalation": False, "reason": "评估完成"}
        return safe_parse_json(result, default_result)


# %%
# ==================== 第十一部分：客服系统主控类 ====================
class CustomerServiceSystem:
    """智能客服系统核心类，整合意图分类、专业处理、质量检查和升级流程"""

    # 质量相关阈值常量
    INTENT_CONFIDENCE_THRESHOLD = 0.6          # 意图置信度低于此值时，进入通用对话而不是直接升级
    QUALITY_SCORE_THRESHOLD = 0.6              # 质检评分低于此值升级（0-1 分数）

    def __init__(self):
        """初始化所有组件并构建工作流状态图"""
        self.classifier = IntentClassifier()                    # 意图分类器
        self.tech_agent = TechSupportAgent()                    # 技术支持Agent
        self.order_agent = OrderServiceAgent()                  # 订单服务Agent
        self.product_agent = ProductConsultAgent()              # 产品咨询Agent
        self.general_agent = GeneralChatAgent()                 # 通用对话Agent
        self.quality_checker = QualityChecker()                 # 质量检查器
        self.current_history = []                               # 当前对话历史（此处保留但未使用，可扩展）
        self.graph = self._build_graph()                        # 构建并编译状态图

    def _build_graph(self):
        """定义并编译客服系统的工作流状态图"""

        # --- 节点函数定义 ---

        def classify_intent(state: CustomerServiceState) -> CustomerServiceState:
            """分类用户意图节点"""
            logger.info("分析用户意图")
            result = self.classifier.classify(state["user_message"])   # 调用分类器
            state["intent"] = result.get("intent", "general_chat")     # 提取意图
            state["confidence"] = result.get("confidence", 0.3)        # 提取置信度
            logger.info(f"意图：{state['intent']} (置信度: {state['confidence']:.2f})")
            return state

        def route_to_agent(state: CustomerServiceState) -> Literal[
            "tech_support", "order_service", "product_consult", "general_chat", "escalate"]:
            """根据意图和置信度决定下一个节点（路由函数）"""
            intent = state["intent"]
            confidence = state["confidence"]

            # 只有当意图是 escalate 且置信度较高，或者明确要求人工时才真正升级
            if intent == "escalate" and confidence >= 0.7:
                return "escalate"

            # 置信度不足时，不是直接升级，而是交给 general_chat 去尝试理解
            if confidence < self.INTENT_CONFIDENCE_THRESHOLD:
                return "general_chat"

            # 正常路由
            if intent == "tech_support":
                return "tech_support"
            elif intent == "order_service":
                return "order_service"
            elif intent == "product_consult":
                return "product_consult"
            else:  # general_chat 或其它情况
                return "general_chat"

        def tech_support_handler(state: CustomerServiceState) -> CustomerServiceState:
            """技术支持处理节点"""
            logger.info("技术支持代理处理中")
            response = self.tech_agent.handle(state["user_message"], state["chat_history"])  # 调用Agent处理
            state["agent_response"] = response   # 保存回复
            return state

        def order_service_handler(state: CustomerServiceState) -> CustomerServiceState:
            """订单服务处理节点"""
            logger.info("订单服务代理处理中")
            response = self.order_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def product_consult_handler(state: CustomerServiceState) -> CustomerServiceState:
            """产品咨询处理节点"""
            logger.info("产品咨询代理处理中")
            response = self.product_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def general_chat_handler(state: CustomerServiceState) -> CustomerServiceState:
            """通用对话处理节点"""
            logger.info("通用对话代理处理中")
            response = self.general_agent.handle(state["user_message"], state["chat_history"])
            state["agent_response"] = response
            return state

        def escalate_handler(state: CustomerServiceState) -> CustomerServiceState:
            """升级节点：当用户明确要求人工时，生成转人工提示"""
            logger.info("升级到人工客服")
            state["needs_escalation"] = True                          # 标记需要升级
            state["escalation_reason"] = "用户明确要求人工服务"        # 记录升级原因
            state["agent_response"] = """非常抱歉，您的问题需要人工客服来处理。
我已经为您转接人工客服，请稍后...

在等待期间，你也可以：
1. 拨打客服热线：400-xxx-xxxx
2. 发送邮件至：support@example.com
3. 工作日 9:00-18:00 在线客服响应更快

感谢您的耐心等待！"""
            return state

        def quality_check(state: CustomerServiceState) -> CustomerServiceState:
            """质量检查节点：评估当前 Agent 的回复质量，必要时触发升级"""
            logger.info("执行质量检查")
            result = self.quality_checker.check(state["user_message"], state["agent_response"])  # 质检

            # 将总分转换为 0-1 范围
            raw_score = result.get("total_score", 0)
            try:
                score = float(raw_score) / 100.0                    # 转换为0-1分数
            except (ValueError, TypeError):
                score = 0.6                                         # 解析失败时给予中等分数
            state["quality_score"] = score

            # 如果是通用对话，放宽升级条件
            if state["intent"] == "general_chat":
                # 只有极低分才升级（低于 0.3）
                if state["quality_score"] < 0.3 and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "通用对话质量过低")
                    state["already_escalated"] = True               # 标记已升级，防止重复
                else:
                    state["needs_escalation"] = False
            else:
                # 其他意图的原有升级逻辑
                if (result.get("needs_escalation", False) or state["quality_score"] < self.QUALITY_SCORE_THRESHOLD) \
                        and not state.get("already_escalated", False):
                    state["needs_escalation"] = True
                    state["escalation_reason"] = result.get("reason", "质量检查未通过")
                    state["already_escalated"] = True

            logger.info(f"质量评分：{state['quality_score']:.2f}")
            return state

        def should_escalate(state: CustomerServiceState) -> Literal["escalate_final", "respond"]:
            """决定是否需要最终升级"""
            if state.get("needs_escalation", False):
                return "escalate_final"
            return "respond"

        def final_escalate(state: CustomerServiceState) -> CustomerServiceState:
            """最终升级节点：在原始回复后追加系统升级提示"""
            original_response = state["agent_response"]
            state["agent_response"] = f"""{original_response}
系统提示：由于此问题可能需要更专业的处理，我们建议您联系人工客服以获得更好的服务。"""
            return state

        def respond(state: CustomerServiceState) -> CustomerServiceState:
            """正常回复节点：什么也不做，直接返回原状态"""
            return state

        # --- 状态图构建 ---
        graph = StateGraph(CustomerServiceState)

        # 添加所有节点
        graph.add_node("classify", classify_intent)          # 意图分类节点
        graph.add_node("tech_support", tech_support_handler) # 技术支持节点
        graph.add_node("order_service", order_service_handler)# 订单服务节点
        graph.add_node("product_consult", product_consult_handler)# 产品咨询节点
        graph.add_node("general_chat", general_chat_handler) # 通用对话节点
        graph.add_node("escalate", escalate_handler)         # 明确升级节点
        graph.add_node("quality_check", quality_check)       # 质量检查节点
        graph.add_node("escalate_final", final_escalate)     # 最终升级节点
        graph.add_node("respond", respond)                   # 正常回复节点

        # 定义流程边
        graph.add_edge(START, "classify")                    # 开始 -> 意图分类
        graph.add_conditional_edges(                         # 根据分类结果路由
            "classify",
            route_to_agent,
            {
                "tech_support": "tech_support",
                "order_service": "order_service",
                "product_consult": "product_consult",
                "general_chat": "general_chat",
                "escalate": "escalate"
            }
        )
        # 各 Agent 处理完后进入质量检查
        graph.add_edge("tech_support", "quality_check")
        graph.add_edge("order_service", "quality_check")
        graph.add_edge("product_consult", "quality_check")
        graph.add_edge("general_chat", "quality_check")
        # escalate 节点跳过质量检查，直接进入回复（因为已经是升级提示）
        graph.add_edge("escalate", "respond")

        # 质量检查后，根据是否需要升级进行路由
        graph.add_conditional_edges(
            "quality_check",
            should_escalate,
            {
                "escalate_final": "escalate_final",          # 需要升级 -> 最终升级节点
                "respond": "respond"                         # 不需要升级 -> 正常回复节点
            }
        )
        graph.add_edge("escalate_final", END)                # 最终升级后结束
        graph.add_edge("respond", END)                       # 正常回复后结束

        return graph.compile()                               # 编译状态图，返回可执行的应用

    def handle_message(self, message: str, chat_history: List[Dict] = None) -> Dict[str, Any]:
        """
        处理用户消息的入口方法。
        返回包含回复、意图、置信度、质量评分和是否升级的字典。
        """
        try:
            logger.info(f"用户消息: {message}")

            # 构建初始状态（除用户消息和聊天历史外，其他字段为空/默认值）
            initial_state = {
                "user_message": message,                     # 用户输入
                "chat_history": chat_history or [],          # 对话历史，默认为空列表
                "intent": "",                                # 意图初始为空
                "confidence": 0.0,                           # 置信度初始为0
                "agent_response": "",                        # 回复初始为空
                "needs_escalation": False,                   # 默认不需要升级
                "escalation_reason": "",                     # 升级原因初始为空
                "quality_score": 0.0,                        # 质量评分初始为0
                "already_escalated": False,                  # 本轮未升级
                "metadata": {"timestamp": datetime.now().isoformat()}  # 记录时间戳
            }

            result = self.graph.invoke(initial_state)        # 执行工作流

            return {
                "response": result["agent_response"],        # 最终回复文本
                "intent": result["intent"],                  # 识别出的意图
                "confidence": result["confidence"],          # 意图置信度
                "quality_score": result["quality_score"],    # 质量评分
                "escalated": result["needs_escalation"]      # 是否升级
            }
        except Exception as e:
            # 全局异常捕获，避免任何未处理错误导致系统崩溃
            logger.error(f"处理消息时发生异常: {e}", exc_info=True)
            return {
                "response": "非常抱歉，系统暂时遇到了一点问题，请稍后再试或联系人工客服。",
                "intent": "escalate",
                "confidence": 0.0,
                "quality_score": 0.0,
                "escalated": True
            }


# %%
# ==================== 第十二部分：演示主程序 ====================
def main():
    """运行多代理智能客服系统的演示，包含预设测试用例和交互式对话"""
    print("=" * 60)                           # 打印分隔线
    print("多代理智能客服系统演示")            # 打印标题
    print("=" * 60)                           # 打印分隔线

    print("\n初始化客服系统...")               # 提示开始初始化
    system = CustomerServiceSystem()          # 创建系统实例，内部完成组件初始化与工作流编译
    print("系统初始化完成！")

    # 预设测试用例：按类别测试不同的对话场景
    test_cases = [
        {
            "category": "技术支持",
            "messages": [
                "我的蓝牙耳机连接不上手机怎么办？",
                "手表充电很慢，是不是坏了？"
            ]
        },
        {
            "category": "订单服务",
            "messages": [
                "帮我查一下订单 ORD001 的物流状态",
                "我的订单什么时候能到？订单号是 ORD002"
            ]
        },
        {
            "category": "产品咨询",
            "messages": [
                "你们有什么智能手表推荐吗？预算1500左右",
                "无线耳机有什么功能？"
            ]
        },
        {
            "category": "人工升级",
            "messages": [
                "我要投诉！这是第三次出问题了！",
                "我想和你们经理谈谈"
            ]
        }
    ]

    # 依次运行预设测试用例
    for test in test_cases:
        print(f"\n{'=' * 60}")
        print(f"测试类别: {test['category']}")   # 打印当前测试类别
        print('=' * 60)

        chat_history = []                        # 每个测试用例重置对话历史

        for message in test["messages"]:         # 遍历该类别下的每一条测试消息
            result = system.handle_message(message, chat_history)  # 调用系统处理

            print("\n客服回复:")
            print(f"{result['response']}")       # 打印回复内容
            print("\n处理信息:")
            print(f"   - 意图: {result['intent']}")          # 打印意图
            print(f"   - 置信度: {result['confidence']:.2f}") # 打印置信度
            print(f"   - 质量评分: {result['quality_score']:.2f}") # 打印质量评分
            print(f"   - 是否升级: {'是' if result['escalated'] else '否'}") # 打印是否升级
            print("-" * 60)

            # 更新对话历史
            chat_history.append({"role": "user", "content": message})        # 添加用户消息
            chat_history.append({"role": "assistant", "content": result['response']}) # 添加助手回复

    # 交互式对话演示：用户可以实时输入问题
    print("\n" + "=" * 60)
    print("交互式对话演示")
    print("=" * 60)
    print("提示: 输入 'quit' 退出")

    chat_history = []                          # 交互模式重置历史

    while True:
        user_input = input("\n您: ").strip()   # 获取用户输入
        if user_input.lower() == 'quit':       # 输入quit退出
            print("\n感谢使用智能客服系统，再见！")
            break
        if not user_input:
            continue                           # 空输入忽略

        result = system.handle_message(user_input, chat_history) # 处理输入
        print(f"\n客服: {result['response']}") # 打印回复

        # 更新对话历史
        chat_history.append({"role": "user", "content": user_input})          # 添加用户消息
        chat_history.append({"role": "assistant", "content": result['response']}) # 添加助手回复


# %%
if __name__ == "__main__":
    main()                                    # 程序入口：直接运行脚本时启动演示