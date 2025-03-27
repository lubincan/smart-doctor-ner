# 服务框架使用Flask
# 导入相关的包
from flask import Flask,request
app = Flask(__name__)
import redis
from neo4j import GraphDatabase
import json
NEO4J_CONFIG = {
	"uri": "bolt://127.0.0.1:7687",
	"auth": ("neo4j", "password"),
	"encrypted": False
}



REDIS_CONFIG = {
	"host": "127.0.0.1",
	"port": 6379
}

driver = GraphDatabase.driver(**NEO4J_CONFIG)

def getNeo4j():
    with driver.session() as ses:
        cypher = "match(e:Ployee) return e.id,e.name"
        result= ses.run(cypher)
        records = [record.data() for record in result]
        print(records)
        #return records[0]['e.name']
        return json.dumps(records,indent=4)

@app.route('/',methods=['GET'])
def hello_world():
    query = request.args.get('query')
    if query :
        r.set("lubc66", str(query).encode('utf-8'))

    result = r.get('lubc66')


    return 'Hello world'+result.decode('utf-8')+getNeo4j()
pool = redis.ConnectionPool(**REDIS_CONFIG)
r =redis.StrictRedis(connection_pool=pool)
r.set("lubc66","我是lucas哈哈哈".encode('utf-8'))
result = r.get('lubc66')
print(result.decode('utf-8'))



if __name__ =='__main__':
    #pass
    app.run(host="0.0.0.0", port=5000)


