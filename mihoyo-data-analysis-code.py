# 1. 游戏行为数据采集SDK代码示例

import json
import time
import uuid
import requests
from hashlib import md5

class GameEventTracker:
    def __init__(self, game_id, server_id, app_key, server_url):
        self.game_id = game_id
        self.server_id = server_id
        self.app_key = app_key
        self.server_url = server_url
        self.session_id = str(uuid.uuid4())
        self.user_id = None
        self.device_info = {}
        
    def init_user(self, user_id, device_info=None):
        """初始化用户信息"""
        self.user_id = user_id
        self.device_info = device_info or {}
        
        # 记录启动事件
        self.track_event("app_launch", {
            "launch_time": int(time.time() * 1000),
            "device_info": self.device_info
        })
        
    def track_event(self, event_name, properties=None):
        """追踪游戏事件"""
        if not self.user_id:
            print("Error: User not initialized")
            return False
            
        event_data = {
            "event": event_name,
            "game_id": self.game_id,
            "server_id": self.server_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": int(time.time() * 1000),
            "properties": properties or {}
        }
        
        # 添加签名
        sign_content = f"{self.user_id}{self.game_id}{self.app_key}{event_data['timestamp']}"
        event_data["sign"] = md5(sign_content.encode()).hexdigest()
        
        # 异步发送数据
        self._send_data(event_data)
        return True
    
    def track_payment(self, order_id, amount, currency, product_id, product_name):
        """追踪支付事件"""
        return self.track_event("payment", {
            "order_id": order_id,
            "amount": amount,
            "currency": currency,
            "product_id": product_id,
            "product_name": product_name,
            "payment_time": int(time.time() * 1000)
        })
        
    def track_level_up(self, level, exp, cost_time):
        """追踪等级提升事件"""
        return self.track_event("level_up", {
            "level": level,
            "exp": exp,
            "cost_time": cost_time
        })
    
    def track_quest_complete(self, quest_id, quest_name, rewards, time_spent):
        """追踪任务完成事件"""
        return self.track_event("quest_complete", {
            "quest_id": quest_id,
            "quest_name": quest_name,
            "rewards": rewards,
            "time_spent": time_spent
        })
        
    def track_gacha(self, gacha_id, gacha_type, cost_item, cost_amount, items_gained):
        """追踪抽卡/抽奖事件"""
        return self.track_event("gacha", {
            "gacha_id": gacha_id,
            "gacha_type": gacha_type,
            "cost_item": cost_item,
            "cost_amount": cost_amount,
            "items_gained": items_gained
        })
    
    def _send_data(self, data):
        """发送数据到服务器(实际实现会使用异步队列)"""
        try:
            requests.post(
                self.server_url,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=2
            )
        except Exception as e:
            # 实际实现中会将失败的请求存入本地缓存，稍后重试
            print(f"Failed to send data: {e}")


# 2. 玩家流失预警模型

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

class ChurnPredictionModel:
    def __init__(self):
        self.model = None
        self.features = None
        
    def prepare_features(self, data_df):
        """准备特征数据"""
        # 活跃度特征
        features = pd.DataFrame()
        features['login_days_last_week'] = data_df['login_days_last_week']
        features['login_days_last_month'] = data_df['login_days_last_month'] 
        features['avg_session_time'] = data_df['avg_session_time']
        features['days_since_last_login'] = data_df['days_since_last_login']
        
        # 游戏进度特征
        features['player_level'] = data_df['player_level']
        features['completion_rate'] = data_df['completion_rate']
        features['achievement_count'] = data_df['achievement_count']
        
        # 社交特征
        features['friend_count'] = data_df['friend_count']
        features['guild_active'] = data_df['guild_active'].astype(int)
        features['social_interaction_weekly'] = data_df['social_interaction_weekly']
        
        # 消费特征
        features['total_spend'] = data_df['total_spend']
        features['days_since_last_purchase'] = data_df['days_since_last_purchase'].fillna(365)
        features['purchase_frequency'] = data_df['purchase_frequency']
        
        # 游戏内容参与
        features['gacha_pulls_weekly'] = data_df['gacha_pulls_weekly']
        features['event_participation'] = data_df['event_participation']
        features['daily_task_completion_rate'] = data_df['daily_task_completion_rate']
        
        return features
    
    def train(self, training_data, label_col='churned'):
        """训练流失预测模型"""
        features = self.prepare_features(training_data)
        self.features = features.columns.tolist()
        
        X = features
        y = training_data[label_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # 使用随机森林分类器
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # 评估模型
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        print("模型性能评估:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.4f}")
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n特征重要性:")
        print(feature_importance.head(10))
        
        return self.model
    
    def predict(self, player_data):
        """预测玩家流失概率"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        features = self.prepare_features(player_data)
        # 确保特征列顺序与训练时一致
        features = features[self.features]
        
        churn_prob = self.model.predict_proba(features)[:, 1]
        
        result = pd.DataFrame({
            'player_id': player_data['player_id'],
            'churn_probability': churn_prob,
            'risk_level': pd.cut(
                churn_prob, 
                bins=[0, 0.3, 0.6, 1], 
                labels=['低风险', '中风险', '高风险']
            )
        })
        
        return result
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise Exception("No model to save")
        
        model_data = {
            'model': self.model,
            'features': self.features
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.features = model_data['features']


# 3. 实时数据处理Flink任务示例

from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings
from pyflink.common.typeinfo import Types
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema

import json
import time

def process_game_events():
    # 设置Flink执行环境
    env = StreamExecutionEnvironment.get_execution_environment()
    env_settings = EnvironmentSettings.new_instance().in_streaming_mode().use_blink_planner().build()
    t_env = StreamTableEnvironment.create(env, environment_settings=env_settings)
    
    # 设置检查点(实现容错)
    env.enable_checkpointing(60000)  # 每60秒做一次检查点
    
    # Kafka连接器属性
    props = {
        'bootstrap.servers': 'kafka:9092',
        'group.id': 'game_events_processor'
    }
    
    # 创建Kafka消费者
    kafka_consumer = FlinkKafkaConsumer(
        'game_events', 
        SimpleStringSchema(),
        properties=props
    )
    
    # 设置从最早的偏移量开始读取
    kafka_consumer.set_start_from_earliest()
    
    # 创建Kafka生产者
    kafka_producer = FlinkKafkaProducer(
        'processed_events',
        SimpleStringSchema(),
        properties=props
    )
    
    # 添加数据源
    game_events = env.add_source(kafka_consumer)
    
    # 处理逻辑
    processed_events = game_events.map(process_event)
    
    # 过滤掉无效事件
    valid_events = processed_events.filter(lambda x: x is not None)
    
    # 添加接收器
    valid_events.add_sink(kafka_producer)
    
    # 执行任务
    env.execute("Game Events Real-time Processing")

def process_event(event_json):
    try:
        event = json.loads(event_json)
        
        # 添加处理时间戳
        event['processing_time'] = int(time.time() * 1000)
        
        # 数据验证
        if 'user_id' not in event or 'event' not in event:
            return None
            
        # 数据丰富
        if event['event'] == 'payment':
            # 添加消费等级分类
            amount = event['properties'].get('amount', 0)
            if amount == 0:
                return None  # 过滤掉金额为0的支付
                
            if amount < 50:
                spend_level = 'low'
            elif amount < 200:
                spend_level = 'medium'
            else:
                spend_level = 'high'
                
            event['properties']['spend_level'] = spend_level
            
        # 加工后的事件
        return json.dumps(event)
    except Exception as e:
        # 错误处理
        print(f"Error processing event: {e}")
        return None


# 4. 数据可视化Dashboard示例(使用Dash)

import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

# 连接数据库(示例使用)
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost:5432/game_analytics')

# 初始化Dash应用
app = dash.Dash(__name__, title="米哈游游戏分析平台")

# 应用布局
app.layout = html.Div([
    html.H1("米哈游游戏数据分析平台"),
    
    html.Div([
        html.Div([
            html.H3("选择游戏"),
            dcc.Dropdown(
                id='game-selector',
                options=[
                    {'label': '原神', 'value': 'genshin'},
                    {'label': '崩坏：星穹铁道', 'value': 'honkai_star_rail'},
                    {'label': '崩坏3', 'value': 'honkai_impact'}
                ],
                value='genshin'
            ),
            
            html.H3("选择时间范围"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=pd.Timestamp('2020-09-01').date(),
                max_date_allowed=pd.Timestamp.now().date(),
                start_date=pd.Timestamp.now().date() - pd.Timedelta(days=30),
                end_date=pd.Timestamp.now().date()
            )
        ], style={'width': '30%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("关键指标"),
            html.Div(id='kpi-cards')
        ], style={'width': '70%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.Div([
            html.H3("日活跃用户(DAU)"),
            dcc.Graph(id='dau-chart')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("收入趋势"),
            dcc.Graph(id='revenue-chart')
        ], style={'width': '50%', 'display': 'inline-block'})
    ]),
    
    html.Div([
        html.H3("用户留存分析"),
        dcc.Graph(id='retention-heatmap')
    ]),
    
    html.Div([
        html.Div([
            html.H3("玩家等级分布"),
            dcc.Graph(id='level-distribution')
        ], style={'width': '50%', 'display': 'inline-block'}),
        
        html.Div([
            html.H3("付费转化率"),
            dcc.Graph(id='conversion-funnel')
        ], style={'width': '50%', 'display': 'inline-block'})
    ])
])

# 回调函数 - 关键指标卡片
@app.callback(
    Output('kpi-cards', 'children'),
    [Input('game-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_kpi_cards(game, start_date, end_date):
    # 实际应用中会从数据库查询
    # 示例数据
    query = f"""
    SELECT 
        SUM(dau) as total_dau,
        SUM(revenue) as total_revenue,
        SUM(new_users) as total_new_users,
        AVG(retention_d1) as avg_d1_retention
    FROM daily_stats
    WHERE game_id = '{game}'
    AND date BETWEEN '{start_date}' AND '{end_date}'
    """
    
    # 模拟数据
    kpi_data = {
        'total_dau': np.random.randint(100000, 500000),
        'total_revenue': np.random.randint(1000000, 5000000),
        'total_new_users': np.random.randint(10000, 50000),
        'avg_d1_retention': np.random.uniform(0.3, 0.5)
    }
    
    return [
        html.Div([
            html.H4("平均日活跃用户"),
            html.H2(f"{int(kpi_data['total_dau'] / 30):,}")
        ], className='kpi-card'),
        
        html.Div([
            html.H4("总收入(元)"),
            html.H2(f"¥{kpi_data['total_revenue']:,}")
        ], className='kpi-card'),
        
        html.Div([
            html.H4("新增用户"),
            html.H2(f"{kpi_data['total_new_users']:,}")
        ], className='kpi-card'),
        
        html.Div([
            html.H4("次日留存率"),
            html.H2(f"{kpi_data['avg_d1_retention']:.1%}")
        ], className='kpi-card')
    ]

# 回调函数 - DAU图表
@app.callback(
    Output('dau-chart', 'figure'),
    [Input('game-selector', 'value'),
     Input('date-range', 'start_date'),
     Input('date-range', 'end_date')]
)
def update_dau_chart(game, start_date, end_date):
    # 生成模拟数据
    date_range = pd.date_range(start=start_date, end=end_date)
    base_dau = 200000 if game == 'genshin' else (150000 if game == 'honkai_star_rail' else 100000)
    
    dau_data = pd.DataFrame({
        'date': date_range,
        'dau': [
            int(base_dau * (1 + np.sin(i/7) * 0.2 + np.random.normal(0, 0.05))) 
            for i in range(len(date_range))
        ]
    })
    
    fig = px.line(dau_data, x='date', y='dau', 
                  title=f"{game.capitalize()} 日活跃用户趋势")
    fig.update_layout(xaxis_title="日期", yaxis_title="活跃用户数")
    
    return fig

# 其他回调函数类似，省略

if __name__ == '__main__':
    app.run_server(debug=True)
