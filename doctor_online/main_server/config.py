
# 设置redis相关的配置信息
REDIS_CONFIG = {
	"host": "127.0.0.1",
	"port": 6379
}

# 设置neo4j图数据库的配置信息
NEO4J_CONFIG = {
	"uri": "bolt://127.0.0.1:7687",
	"auth": ("neo4j", "password"),
	"encrypted": False
}

# 设置句子相关服务的请求地址
model_serve_url = "http://0.0.0.0:5001/v1/recognition/"

# 设置服务的超时时间
TIMEOUT = 2

# 设置规则对话的模板加载路径
reply_path = "./reply.json"

# 用户对话信息保存的过期时间
ex_time = 36000

# 标签结构列表
LABEL_STRUCTURE = [

	{
		"泛娱乐": [
			"明星",
			"时尚",
			"游戏",
			"影视",
			"音乐",
			"美妆"
		]
	},
	{
		"游戏": [
			"LOL",
			"王者农药",
			"吃鸡"
		],
		"影视": [
			"喜剧",
			"综艺",
			"科幻",
			"恐怖"
		],
		"音乐": [
			"摇滚乐",
			"民谣",
			"Rap",
			"流行乐"
		]
	}
]