import sqlite3
import traceback
from typing import Union
import time
import requests  # 保留 requests 以防万一，虽然 dashscope 内部会处理
import pandas as pd
from sqlalchemy import create_engine, inspect, Table, MetaData, select, func, text

# ==========================================
# 🚀 1. 引入千问 SDK
# ==========================================
import dashscope
from dashscope import Generation

# ⚠️ 注意：请在这里填入你的千问 API Key
# 你可以在阿里云控制台 -> DashScope -> API Key 管理 中获取
dashscope.api_key = "sk-0c18954972f8450c95480d5c3c0a82f6"


# ==========================================
# 🤖 2. 定义千问调用函数
# ==========================================
def ask_qwen(prompt, model_name='qwen-plus'):
    """
    调用千问模型生成回答
    """
    try:
        response = Generation.call(
            model=model_name,
            prompt=prompt
        )
        # 检查调用是否成功
        if response.status_code == 200:
            return response.output.text
        else:
            print(f"❌ API 调用失败: {response.code} {response.message}")
            return None
    except Exception as e:
        print(f"❌ 发生异常: {e}")
        return None


# ==========================================
# 💾 3. 数据库解析类 (保持不变)
# ==========================================
class DBParser:
    def __init__(self, db_url: str) -> None:
        if 'sqlite' in db_url:
            self.db_type = 'sqlite'
        elif 'mysql' in db_url:
            self.db_type = 'mysql'

        # 连接数据库
        self.engine = create_engine(db_url, echo=False)
        self.conn = self.engine.connect()
        self.db_url = db_url

        # 查看表名
        self.inspector = inspect(self.engine)
        self.table_name = self.inspector.get_table_names()

        self._table_fields = {}
        self.foreign_keys = []
        self._table_sample = {}

        for table_name in self.table_name:
            print("Table ->", table_name)

            # 获取字段信息
            table_instance = Table(table_name, MetaData(), autoload_with=self.engine)
            table_columns = self.inspector.get_columns(table_name)

            # 初始化字典
            self._table_fields[table_name] = {x['name']: x for x in table_columns}

            # 统计字段信息
            for column_meta in table_columns:
                column_instance = getattr(table_instance.columns, column_meta['name'])

                # 统计 unique
                query = select(func.count(func.distinct(column_instance)))
                distinct_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['distinct'] = distinct_count

                # 统计 most frequency value
                field_type = str(self._table_fields[table_name][column_meta['name']]['type'])
                if 'text' in field_type.lower() or 'char' in field_type.lower():
                    query = (
                        select(column_instance, func.count().label('count'))
                        .group_by(column_instance)
                        .order_by(func.count().desc())
                        .limit(1)
                    )
                    top1_value = self.conn.execute(query).fetchone()[0]
                    self._table_fields[table_name][column_meta['name']]['mode'] = top1_value

                # 统计 missing
                query = select(func.count()).filter(column_instance == None)
                nan_count = self.conn.execute(query).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['nan_count'] = nan_count

                # 统计 max/min
                self._table_fields[table_name][column_meta['name']]['max'] = \
                self.conn.execute(select(func.max(column_instance))).fetchone()[0]
                self._table_fields[table_name][column_meta['name']]['min'] = \
                self.conn.execute(select(func.min(column_instance))).fetchone()[0]

                # 随机值
                query = select(column_instance).limit(10)
                random_value = self.conn.execute(query).all()
                random_value = [str(x[0]) for x in random_value if x[0] is not None]
                self._table_fields[table_name][column_meta['name']]['random'] = list(set(random_value))[:3]

            # 处理外键
            self.foreign_keys += [
                {
                    'constraint_name': table_name,
                    'constrained_columns': x['constrained_columns'],
                    'referred_columns': x['referred_columns'],
                }
                for x in self.inspector.get_foreign_keys(table_name)
            ]

            # 获取表样例
            query = select(table_instance)
            self._table_sample[table_name] = pd.DataFrame([self.conn.execute(query).fetchone()])
            self._table_sample[table_name].columns = [x['name'] for x in table_columns]

    def get_table_fields(self, table_name) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._table_fields[table_name]).T

    def get_data_relations(self) -> pd.DataFrame:
        return pd.DataFrame(self.foreign_keys)

    def get_table_sample(self, table_name) -> pd.DataFrame:
        return self._table_sample[table_name]

    def check_sql(self, sql) -> Union[bool, str]:
        try:
            self.conn.execute(text(sql))
            return True, 'ok'
        except Exception as e:
            err_msg = traceback.format_exc()
            return False, err_msg

    def execute_sql(self, sql) -> list:
        '''运行SQL'''
        result = self.conn.execute(text(sql))
        return list(result)


# ==========================================
# 💬 4. 交互式问答入口 (修改为调用 ask_qwen)
# ==========================================
def chat_with_db(parser):
    print("\n" + "=" * 30)
    print("💬 已启动数据库智能问答助手 (千问版)")
    print("📝 输入 'quit' 或 'exit' 退出对话")
    print("=" * 30)

    while True:
        user_question = input("\n🙋 请输入你的问题: ").strip()

        if user_question.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break

        if not user_question:
            continue

        try:
            # 1. 准备背景知识
            all_tables_info = "数据库中存在的表有: " + ", ".join(parser.table_name)

            # 2. 构造提示词 (让 AI 生成 SQL)
            system_prompt = f"""
            你是一个精通 SQL 的数据分析师。
            请根据以下数据库信息，将用户的问题转换为 SQLite 语法的 SQL 查询语句。
            只需要输出 SQL 代码，不要包含 Markdown 格式（如 ```sql），不要包含解释。

            {all_tables_info}
            """

            # 3. 让 AI 生成 SQL
            print("🤖 正在思考并生成 SQL...")
            generated_sql = ask_qwen(system_prompt + "\n用户问题: " + user_question)

            if not generated_sql:
                print("❌ AI 生成失败，请检查 API Key 或网络连接。")
                continue

            # 清理可能存在的 markdown 标记
            generated_sql = generated_sql.replace("```sql", "").replace("```", "").strip()
            print(f"🔍 生成的 SQL: {generated_sql}")

            # 4. 执行 SQL
            sql_result = parser.execute_sql(generated_sql)

            # 5. 让 AI 分析结果
            print("🤖 正在分析数据并生成回答...")
            answer_prompt = f"""
            你是一个数据分析师助手。
            用户的问题是：{user_question}
            我们查询数据库得到的原始结果是：{sql_result}

            请根据这个结果，用自然、流畅的中文回答用户的问题。
            """
            final_answer = ask_qwen(answer_prompt)

            if final_answer:
                print(f"\n✨ 最终回答: {final_answer}")
            else:
                print("❌ AI 分析结果失败。")

        except Exception as e:
            print(f"❌ 出错了: {e}")


# --- 运行交互 ---
if __name__ == "__main__":
    # 实例化解析器
    parser = DBParser('sqlite:///./chinook.db')

    # 启动聊天
    chat_with_db(parser)