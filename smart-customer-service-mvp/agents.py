import os, re
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ---------- Agent 逻辑 ----------
def product_agent(query: str) -> str:
    sys = "你是专业的产品顾问。根据问题介绍产品功能、价格、库存等。简洁准确。"
    res = llm.invoke([SystemMessage(content=sys), HumanMessage(content=query)])
    return res.content

def after_sales_agent(query: str) -> str:
    sys = "你是售后专员，处理退换货、退款、物流查询。礼貌安抚，给出清晰流程。"
    res = llm.invoke([SystemMessage(content=sys), HumanMessage(content=query)])
    return res.content

def emotion_agent(query: str) -> str:
    sys = "用户情绪激动。请先真诚道歉，安抚情绪，然后引导到具体问题解决。语气温和。"
    res = llm.invoke([SystemMessage(content=sys), HumanMessage(content=query)])
    return res.content

def general_agent(query: str) -> str:
    sys = "你是通用客服助手，回答问候、感谢和其他不属于产品/售后的问题。友好简洁。"
    res = llm.invoke([SystemMessage(content=sys), HumanMessage(content=query)])
    return res.content

# ---------- 简易关键词路由 + 情绪检测 ----------
def detect_emotion(text: str) -> bool:
    # 简单规则：包含强烈负面词则触发情绪Agent
    trigger_words = ["投诉", "差劲", "垃圾", "气死", "太慢", "退款", "欺骗", "骗子"]
    return any(w in text for w in trigger_words)

def route_intent(text: str) -> str:
    if detect_emotion(text):
        return "emotion"
    if any(kw in text for kw in ["产品", "价格", "型号", "配置", "功能", "库存", "参数"]):
        return "product"
    if any(kw in text for kw in ["退", "换", "修", "物流", "订单", "发货", "退款", "快递"]):
        return "after_sales"
    return "general"

async def process_message(message: str, session_id: str) -> str:
    intent = route_intent(message)
    if intent == "product":
        return product_agent(message)
    elif intent == "after_sales":
        return after_sales_agent(message)
    elif intent == "emotion":
        # 情绪安抚后再尝试解决问题（简单组合）
        emotion_reply = emotion_agent(message)
        follow_up = after_sales_agent(message) if any(kw in message for kw in ["退", "换", "修", "物流", "订单", "发货", "退款", "快递"]) else ""
        return emotion_reply + ("\n\n关于您的问题：" + follow_up if follow_up else "")
    else:
        return general_agent(message)