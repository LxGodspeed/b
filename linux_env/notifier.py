import datetime
import requests

class ExperimentNotifier:
    """实验结果通知服务"""
    
    def __init__(self, token=None, api_url="http://www.pushplus.plus/send"):
        """初始化推送服务"""
        self.token = token
        self.api_url = api_url
        self.enabled = bool(token)
        
        if self.enabled:
            print(f"通知服务已启用，令牌: {token[:4]}...{token[-4:]}")
        else:
            print("通知服务未启用")
    
    def send_notification(self, title, content, template="html"):
        """发送推送通知"""
        if not self.enabled:
            return False
            
        try:
            data = {"token": self.token, "title": title, "content": content, "template": template}
            response = requests.post(self.api_url, json=data)
            result = response.json()
            
            if result.get("code") == 200:
                print(f"通知推送成功: {title}")
                return True
            else:
                print(f"通知推送失败: {result.get('msg', '未知错误')}")
                return False
        except Exception as e:
            print(f"通知推送出错: {e}")
            return False
    
    def send_discovery_notification(self, identifier, key_value, public_data, value):
        """发送重要发现通知"""
        title = f"实验发现重要结果! - {identifier}"
        
        # 使用更简洁的HTML格式
        content = f"""
        <div style="padding:10px;background:#f8f9fa;border-radius:5px">
            <h2 style="color:#e74c3c;text-align:center">🎉 实验重要发现! 🎉</h2>
            <div style="margin:10px 0;padding:10px;background:#fff;border-left:4px solid #2980b9">
                <p><strong>标识符:</strong> {identifier}</p>
                <p><strong>密钥值:</strong> {key_value}</p>
                <p><strong>公开数据:</strong> {public_data}</p>
                <p><strong>关联值:</strong> {value}</p>
                <p><strong>发现时间:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            <p style="color:#7f8c8d;font-size:12px;text-align:center">由实验自动监测系统推送</p>
        </div>
        """
        
        return self.send_notification(title, content) 