import threading
import json

from DatabaseConfig import DatabaseConfig
from Database import Database
from BotServer import BotServer
from Preprocess import Preprocess
from IntentModel import IntentModel
from NerModel import NerModel
from FindAnswer import FindAnswer
DB_HOST = "127.0.0.1"
DB_USER = "root"
DB_PASSWORD="1234"
DB_NAME = "test"

def DatabaseConfig():
    global DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
    
p = Preprocess(word2index_dic = '../train_tools/dict/chatbot_dict.bin', userdic = '../utils/user_dic.tsv')
intent = IntentModel(model_name = '../models/intent/intent_model.h5', proprocess = p)
ner= NerModel(model_name1 = '../models/ner/ner_model.h5', proprocess = p)

def to_client(conn, addr, params):
    db = params['db']
    
    try:
        db.connect()  # 디비 연결

        # 데이터 수신
        read = conn.recv(2048)  # 수신 데이터가 있을 때 까지 블로킹
        print('===========================')
        print('Connection from: %s' % str(addr))
        if read is None or not read:
            # 클라이언트 연결이 끊어지거나, 오류가 있는 경우
            print('클라이언트 연결 끊어짐')
            exit(0)


        # json 데이터로 변환
        recv_json_data = json.loads(read.decode())
        print("데이터 수신 : ", recv_json_data)
        query = recv_json_data['Query']
        # 의도 파악
        intent_predict = intent.predict_class(query)
        intent_name = intent.labels[intent_predict]

        # 개체명 파악
        ner_predicts = ner.predict(query)
        ner_tags = ner.predict_tags(query)


        # 답변 검색
        try:
            f = FindAnswer(db)
            answer_text, answer_image = f.search(intent_name, ner_tags)
            answer = f.tag_to_word(ner_predicts, answer_text)

        except:
            answer = "죄송해요 무슨 말인지 모르겠어요. 조금 더 공부 할게요."
            answer_image = None

        send_json_data_str = {
            "Query" : query,
            "Answer": answer,
            "AnswerImageUrl" : answer_image,
            "Intent": intent_name,
            "NER": str(ner_predicts)
        }
        message = json.dumps(send_json_data_str)
        conn.send(message.encode())

    except Exception as ex:
        print(ex)
        
    finally:
        if db is not None: # db 연결 끊기
            db.close()
        conn.close()


if __name__ == '__main__':
    
    db = Database(
        host=DB_HOST, user=DB_USER, password=DB_PASSWORD, db_name=DB_NAME
    )
    print("DB 접속")

    port = 5050
    listen = 100

    # 봇 서버 동작
    bot = BotServer(port, listen)
    bot.create_sock()
    print("bot start")

    while True:
        conn, addr = bot.ready_for_client()
        params = {
            "db": db
        }

        client = threading.Thread(target=to_client, args=(
            conn,
            addr,
            params
        ))
        client.start()
        