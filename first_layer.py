from groq import Groq
from openai import OpenAI
import os

class FirstLayer():
    def __init__(self):
        self.prompt = """<role>You are an advanced AI assistant specializing in classifying user messages for a customer service system of a financial company named FE Credit. Your task is to accurately categorize each incoming message into one of three main classes: Special Treatment, FAQ, or Flow. Then, you must further classify the message into a specific subclass within the chosen main class.</role>

Main Classes and their Subclasses:
<class>
1. Special Treatment:
   - Customer Complaints or Frustration: customer's complaint in general
   - Pending Complaint Handling: customer already sent complaint but was not handled
   - Frequent Contacts Due to Complaint Delays: customer complains frequently in the past few days
   - Same-Day Issue Reporting: customer contact more than once in the same day
   - Request to Speak to a Live Agent

2. FAQ:
   - Support Channels: question about contact method including online, in-person or mobile
   - Payment Channels: question about debt payment methods, names of beneficiary bank or account number
   - CIC Information: question about groups of debt, credit score and non-performing loan
   - Product Information: question about categories of loans and credit cards
   - Loan Registration - Credit Card Registration: Procedure, conditions and or hotline support when getting loans, opening credit cards
   - Issue Resolution: question about disbursement, location to withdraw cash or issues in general
   - Insurance: question about insurance
   - Payment Adjustment/Refund: question about refund and payment adjustment in case of wrong payment
   - Card Activation: question about activating cards
   - Rewards Points: question about reward points feature

3. Flow:
   - Loan Status Inquiry: Loan status and loan transaction history
   - Card Status Inquiry: Card status and card transaction history
</class>

<format>
Your response should always be in the format: "Main Class - Subclass" /n
For example: "Special Treatment - Customer Complaints or Frustration" or "FAQ - Payment Channels"
</format>

Analyze the content, context, and tone of each message carefully. Pay attention to KEYWORDS, phrases, and implied meanings that can help you determine the most appropriate classification. Consider the following guidelines:

1. Special Treatment: Messages that indicate customer dissatisfaction, urgent issues, or requests for personalized attention.
2. FAQ: General inquiries about products, services, or processes that can be answered with standardized information, including loan and card service for people do not have their own card or loan.
3. Flow: Specific requests related to loan or card services for people who already have card or loan that may require accessing or modifying customer accounts.

If a message could potentially fit into multiple categories, choose the most relevant and specific classification based on the primary intent of the user's message.

Remember, your goal is to provide accurate and consistent classifications to ensure that customer inquiries are routed to the appropriate department or handled with the right level of priority."""
        self.history = [
            {"role": "user", "content": """Tôi cảm thấy không hài lòng với dịch vụ"""},
            {"role": "assistant", "content": """Special Treatment - Customer Complaints or Frustration"""},
            {"role": "user", "content": """Tôi đang chờ giải quyết khiếu nại của mình"""},
            {"role": "assistant", "content": """Special Treatment - Pending Complaint Handling"""},
            {"role": "user", "content": """Tôi đã liên hệ nhiều lần về khiếu nại của mình trong những ngày qua nhưng chưa được giải quyết."""},
            {"role": "assistant", "content": """Special Treatment - Frequent Contacts Due to Complaint Delays"""},
            {"role": "user", "content": """Hôm nay tôi đã báo cáo vấn đề nhưng chưa được giải quyết"""},
            {"role": "assistant", "content": """Special Treatment - Same-Day Issue Reporting"""},
            {"role": "user", "content": """Tôi muốn nói chuyện với nhân viên chăm sóc khách hàng"""},
            {"role": "assistant", "content": """Special Treatment - Request to Speak to a Live Agent"""},
            {"role": "user", "content": """Có những kênh hỗ trợ nào"""},
            {"role": "assistant", "content": """FAQ - Support Channels"""},
            {"role": "user", "content": """Công ty em địa chỉ ở đâu thế"""},
            {"role": "assistant", "content": """FAQ - Support Channels"""},
            {"role": "user", "content": """Tôi có thể thanh toán qua những kênh nào"""},
            {"role": "assistant", "content": """FAQ - Payment Channels"""},
            {"role": "user", "content": """Cập nhật lại nhóm nợ của anh được không"""},
            {"role": "assistant", "content": """FAQ - CIC Information"""},
            {"role": "user", "content": """Bạn có thể cung cấp thông tin CIC không?"""},
            {"role": "assistant", "content": """FAQ - CIC Information"""},
            {"role": "user", "content": """Cho tôi biết về các sản phẩm của bạn"""},
            {"role": "assistant", "content": """FAQ - Product Information"""},
            {"role": "user", "content": """Tôi cần vay tiền thì phải làm thủ tục gì"""},
            {"role": "assistant", "content": """FAQ - Loan Registration - Credit Card Registration"""},
            {"role": "user", "content": """Tôi muốn đăng kí thẻ thì phải làm thế nào"""},
            {"role": "assistant", "content": """FAQ - Loan Registration - Credit Card Registration"""},
            {"role": "user", "content": """Tôi có vấn đề cần được giải quyết"""},
            {"role": "assistant", "content": """FAQ - Issue Resolution"""},
            {"role": "user", "content": """Có những loại bảo hiểm nào"""},
            {"role": "assistant", "content": """FAQ - Insurance"""},
            {"role": "user", "content": """Làm sao để điều chỉnh thanh toán hoặc yêu cầu hoàn tiền"""},
            {"role": "assistant", "content": """FAQ - Payment Adjustment/Refund"""},
            {"role": "user", "content": """Làm sao để kích hoạt thẻ"""},
            {"role": "assistant", "content": """FAQ - Card Activation"""},
            {"role": "user", "content": """Làm thế nào để kiểm tra và quy đổi điểm thưởng"""},
            {"role": "assistant", "content": """FAQ - Rewards Points"""},
            {"role": "user", "content": """Cho tôi xem lịch trả nợ của khoản vay"""},
            {"role": "assistant", "content": """Flow - Personal Loan Information"""},
            {"role": "user", "content": """Cho tôi xem lịch sử giao dịch của thẻ"""},
            {"role": "assistant", "content": """Flow - Card Information Inquiry: tra cứu thông tin thẻ và lịch sử thanh toán"""}
        ]
        

        self.client = OpenAI(api_key = os.environ.get("OPENAI_API_KEY"))

    def get_response(self,question):
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": self.prompt
            },
            *self.history,
            {
                "role": "user",
                "content": question
            }
            # {
            #     "role": "assistant",
            #     "content": "Flow - Card Information Inquiry: tra cứu thông tin thẻ và lịch sử thanh toán"
            # },
            # {
            #     "role": "user",
            #     "content": "Giải thích vì sao lại đưa ra lựa chọn như trên"
            # }

        ],
        temperature=0.0,
        max_tokens=1024,
        top_p= 0.5,
        stop=None,
        stream=False
    )
        return response.choices[0].message.content