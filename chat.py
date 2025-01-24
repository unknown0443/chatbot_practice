import os
import google.generativeai as genai
import json
from get_namuwiki_docs import load_namuwiki_docs_selenium
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

# with open("key.json", 'r') as file:
#     data = json.load(file)
    
# gemini_api_key = data.get("gemini-key")

# TODO: 아래 YOUR-HUGGINGFACE-API-KEY랑 OUR-GEMINI-API-KEY에 자기꺼 넣기
if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "u r key"    
gemini_api_key = "u r"

genai.configure(api_key=gemini_api_key)

# gemini 모델 로드 
def load_model():
    with st.spinner("모델을 로딩하는 중..."):
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Model loaded...")
    return gemini_model

# 임베딩 로드
def load_embedding():
    with st.spinner("임베딩을 로딩하는 중..."):
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    print("Embedding loaded...")
    return embedding

# Faiss vector DB 생성
def create_vectorstore(topic):     
    with st.spinner("나무위키에서 문서를 가져오는 중..."):
        # text = load_namuwiki_docs_selenium(topic)        
        # st.write(f"찾은 문서 예시:\n{text[:100]}")
#어마어마한 정보 
#줄바꿈해도 스트링 해도
#유지
   
    # with open("documents.txt") as f:
        # text = f.readline

        text = """
최근 변경
최근 토론
특수 기능
여기에서 검색
밈(인터넷 용어)
최근 수정 시각: 2025-01-24 00:04:53
167
편집
토론
역사


분류인터넷 밈
다른 뜻 아이콘  인터넷 밈은(는) 여기로 연결됩니다. 나무위키의 인터넷 밈 등재 규정에 대한 내용은 나무위키:편집지침/등재 기준 문서의 인터넷 밈 부분을 참고하십시오.
1. 개요
2. 문화적 특징
2.1. 인식
2.1.1. 밈의 수명 및 인싸개그 관련
2.1.2. 정치적 측면 관련
3. 역사
4. 국가별 밈 목록
4.1. 대한민국
4.2. 해외
4.2.1. 일본
5. 해외에서 파생된 밈
5.1. Geometry Dash 계열
5.2. Grand Theft Auto 시리즈 계열
5.3. 네모바지 스폰지밥 계열
5.4. 드래곤볼 계열
5.5. 릭 앤 모티
5.6. 마다가스카의 펭귄 계열
5.7. 마리오 시리즈 계열
5.8. 마인크래프트 계열
5.9. 별의 커비 시리즈 계열
5.10. 소닉 더 헤지혹 시리즈 계열
5.11. 팩맨 시리즈 계열
5.12. 슈렉 시리즈 계열
5.13. 스타워즈 계열
5.14. 심슨 가족 계열
5.15. 유튜브 리와인드 2018
5.16. 이니셜 D 계열
5.17. 포켓몬스터 계열
5.18. 레인보우 식스 시즈 계열
5.19. 팀 포트리스 2 계열
6. 시기별 유행 밈
6.1. 대한민국
6.1.1. 1997년, 1999년
6.1.2. 2000년대
6.1.3. 2010년
6.1.4. 2011년
6.1.5. 2012년
6.1.6. 2015년
6.1.7. 2016년
6.1.8. 2017년
6.1.9. 2018년
6.1.10. 2019년
6.1.11. 2020년
6.1.12. 2021년
6.1.13. 2022년
6.1.14. 2023년
6.1.15. 2024년
6.1.16. 2025년
6.2. 해외
6.2.1. 1996년
6.2.2. 2000년대
6.2.3. 2012년
6.2.4. 2013년
6.2.5. 2016년
6.2.6. 2017년
6.2.7. 2018년
6.2.8. 2019년
6.2.9. 2020년
6.2.10. 2021년
6.2.11. 2022년
6.2.12. 2023년
6.2.13. 2024년
6.2.14. 2025년
7. 기타 목록
8. 여담
9. 관련 문서
1. 개요[편집]
인터넷 밈(Internet meme)은 인터넷 커뮤니티나 SNS 등지에서 퍼져나가는 여러 문화의 유행과 파생·모방의 경향, 또는 그러한 창작물이나 작품의 요소를 총칭하는 용어이다.

본래 1976년 동물학자 리처드 도킨스가 저서 이기적 유전자에서 처음 제시한 학술 용어인 '밈(meme)'에서 파생된 개념으로, 밈은 마치 인간의 '유전자(진, gene)'와 같이 '자기복제적 특징을 갖고, 번식해 대를 이어 전해져 오는 종교나 사상, 이념 같은 정신적 사유'를 의미했다. 이것이 '패러디되고 변조되며 퍼지는 작품 속 문화 요소'라는 의미로 확대된 것은 90년대 후반에서 2000년대 초반으로, 인터넷이 보급된 뒤 폭발적으로 늘어나는 새로운 방식의 문화 전파 현상을 도킨스의 표현을 빌려 나타낸 것이다.
2. 문화적 특징[편집]

Coffin Dance
(조회수 약 4.4억회)

Camila Cabello - Havana ( cover by Donald Trump )
(조회수 약 1.3억회)
대체로 특정 요인에 따른 유행 전반을 통칭하는 개념으로 유행어와 비슷한 부분이 많다. 다만 밈은 언어에 국한되지 않고 사진이나 영상 속 요소 등 다양한 미디어를 넘나든다는 차이가 있다. 이런 점을 고려하면 우리말로 적당히 대체할 만한 용어론 '짤방' 내지 '필수요소'가 있는데 뜻이 정확하게 일치하지는 않는다. 내가 고자라니를 예로 들면, '필수요소'가 '내가 고자라니'라는 소스(Source) 자체에 결속된 의미라면, 밈은 이걸 가지고 합성을 해서 나온 다양한 심영물 등 소재를 활용한 형태들을 전부 포함한다. 이 경우 짤방이 밈에 대응되는 것이라고 볼 수 있으나 짤방은 이미지나 움직이는 이미지를 매체로 하는 것에 한정되지만 밈은 원칙적으로 매체의 한정이 없다.[1]

또 다른 결정적 차이는 밈은 특정한 소스뿐 아니라 그 소스를 사용하는 방법 역시 규격화되어 있다는 점이다. 밈이란 게 원래 문화 현상을 유전 현상에 빗대기 위해 만든 단어임을 생각하면 당연히 유추할 수 있는 것이기는 하다. 예컨대 '내가 고자라니'의 곶 부분만 뗴어내서 노래를 부르게 만들었다면 필수요소 관점에선 심영을 포함하게 되나 밈 관점에서는 인간 관악기 밈의 하위 분류가 되지 '내가 고자라니'의 하위 분류가 되지는 않는다. 상황에 맞게 사용된 것이 아니기 때문이다. 나무위키에서 통용되었던[2] 밈 중 하나인 나무위키 암묵의 룰 문서가 이런 밈의 특징을 잘 짚어내고 있다.

원칙적인 차이점을 하나 더 들면 한국의 필수요소는 사진 및 영상물 중심이고 주로 합필갤에서 좌지우지되는 인터넷 문화 요소인 것과 비교해, 밈은 사진 및 영상물뿐만 아니라 유행어 등 훨씬 더 포괄적이고 특정 사이트에 좌지우지되는 게 아니라 수많은 웹사이트들의 통합적인 유행 요소 전체이다. 이와 같은 밈의 의미는 애초에 4chan에서 비롯되었고, 2000년대를 이끌던 밈들도 십중팔구가 4chan에서 만들어졌다. 두 단어가 차이가 있긴 하지만 한국에서는 합필갤의 인기가 식어감과 동시에 '밈'이라는 단어가 사용되기 시작하던 게 겹쳐서 사실상 같은 의미를 내포하는 단어가 세대 교체를 이루는 흐름이 되었다.

과거 상당수의 밈은 그림 파일 내지 GIF로 이루어져 한국의 짤방 같은 개념에 대체로 대응되었으나, 애당초 이미지에 한정되었던 짤방개념과 달리 밈은 모든 매체로 이루어질 수 있었고, 기술의 발달로 인터넷에서 멀티미디어를 활용하는 것이 점점 더 용이해지면서 현재는 더 포괄적으로 그냥 유행하는 대부분의 것을 밈이라고 불러도 무방하다.[3] 인터넷에 나도는 기억하기 쉽고 병맛이거나 중독성 있는 대상이라면 무엇이든지 밈으로 등극할 가능성이 있다. 주요 밈의 유래를 따져 보면 밈이 밈으로 등극하는 규칙은 거의 없다고 해도 될 정도다.[4] 병맛이나 중독성이 그다지 있지 않더라도 누군가가 이것을 발견하고 편집해서 올렸는데 그 편집이 중독성있거나 병맛이라면 그것도 밈이 될 수 있다.

게다가 밈이 재조합되어 새로운 밈을 만들어내는 경우도 있다. 하츠네 미쿠 보컬로이드에 넣어서 만든 Nyanyanyanyanyanyanya!가 어느 정도 밈으로 퍼지다가 이게 팝타르트 고양이를 만나서 새로 만들어진 Nyan Cat[5]이라든지, 이미 인터넷에서 컬트적인 인기를 끌고 있던 하프라이프의 막장 한국어 더빙이 사진 자료와 음향 합성을 만나면서 만들어진 장비를 정지합니다 같은 경우가 있다.

특정 공식 작품(게임, 애니메이션 등)에 대해 팬들이 드립을 치고 그게 밈으로 발전하는 경우, 후속작이나 업데이트로 그 밈이 공식화되는 경우가 있는데, 이를 'Ascended Meme'이라고 한다. 일본에서는 'まさかの公式(설마했던 공식)'으로 불리며, 공식 설정이 동인 설정을 자주 참고하고 가끔은 역수입까지 하는 경향이 있는 함대 컬렉션과 아이돌마스터 시리즈에서 자주 볼 수 있다. 인터넷 필수요소의 변화를 잘 표현한 짤로는 이것 참조.

참고로 밈이라고 꼭 대중에게 유명하라는 법은 없다. 따라서 특정 집단 내에서도 당연히 밈이 존재한다. 영화 팬덤, 연예인 팬덤, 드라마 팬덤, 커뮤니티 등 이 안에 그들만의 밈은 존재한다. 예를 들면 해외 스타워즈 시리즈 레딧 팬덤이 만든 prequel_memes같은 경우가 있다. 가끔씩 유명인들이 자기 별명/유행어/밈을 직접 리뷰하는 경우도 있다.[6]

모든 밈은 유통기한 겸 수명이 존재한다. 음지에서 양지로 올라올수록, 소수집단에서 대중으로 쓰이는 계층이 넘어갈수록 밈의 수명이 줄어든다. # 유튜브나 틱톡 한정으로 밈의 제작 난이도 역시 수명을 결정짓는 아주 중요한 요소이다. 만들기 쉬울수록 수명이 줄어들고 만들기 어려울수록 수명이 늘어난다. 빅맥송과 제로투 댄스가 이 두 가지 경우를 잘 보여준다.[7]

유튜브에서 불규칙적인 주기로 유행하는 인터넷 밈들은 이름없는 무명의 유튜버들이 크게 성장하여 인지도를 쌓을 수 있는 일종의 기회로 작용하기도 한다. 실제로도 이런 방식으로 급성장한 유명 유튜버들이 드물지 않으며 아무리 인지도가 낮은 유튜버라도 인터넷 밈 영상을 하나라도 만들어 올리면 유튜브 알고리즘에 탑승하여 구독자를 크게 올릴 수 있는 가능성이 열린다. 하지만 뭐든지 과도하면 안 좋듯 이미 한번 만든것을 여러번 우려먹거나 스스로의 역량부족 등으로 인해 유입된 시청자들의 니즈를 맞춰주지 못한다면 모든 것이 원점으로 돌아갈 수도 있으니 주의하는 것이 좋다.

밈들은 유쾌하고 코믹한, 아무리 적어도 블랙 코미디 정도의 유머를 띄고 있다. 본질적으로 밈은 장난이기 때문이다. 정말 진지하거나 슬픈 의도의 이야기라면 당연히 장난 식으로 퍼지지 않는다. 악인의 죽음은 밈으로 퍼질 수 있지만, 선량한 사람의 죽음 같은 일은 밈으로 퍼져서 안 되며 그럴 수도 없다.

스마트폰 보급과 소셜미디어 발달 이후 청소년, 청년들 사이에서 틱톡, 인스타그램의 스토리와 릴스, 트위터 게시글 등을 통해 알려지는 감상하는 입장에서 재미가 있다는 이유로 쉽게 너도 나도 따라하면서 공유하고 응용하며 확산되기 쉽지만, 이와 달리 소셜 미디어를 사용하지 않는 사람들이나 중년층 이상의 일반인이 보기에는 그저 화면 속에서 젊은 사람들이 춤추거나 노래하는 영상에 불과한 부분도 있다. 재미를 포함한 흥미로움이나 몰입의 기준이 절대적인 것이 아니라 사람마다 상대적이라는 점에서 유의해야 할 부분.

익명 커뮤니티의 불건전하거나 반사회적 내용의 밈들이 인터넷 상에서 공공연하게 쓰이고 확산되는 것 같지만, 실은 끼리끼리 모인 해당 커뮤니티 내에서만 통용될 뿐이라는 그저 비슷한 개인들간의 착각, 확증편향에 해당하는 소수집단의 한계적인 면도 있다. 게임, 영화, 대중가요 등을 위시한 탑티어 아이돌이나 배우, 인플루언서, 인터넷 방송인, 프로게이머들에 관한 내용들도 마찬가지다.
2.1. 인식[편집]
2.1.1. 밈의 수명 및 인싸개그 관련[편집]
해외에서는 밈을 좋아하는 몇몇 유저들은 특정 밈을 멋대로 설명하는 행위를 싫어하는 경향이 강하다. '전후 사정도 모르고 설명으로만 밈을 이해했을 때 재미가 반감된다', '자신들이 좋아하는 밈들이 대중에게 널리 퍼지면 점점 재미가 없어진다' 등. 이렇게 설명을 듣고 밈을 배운 사람들은 노미(normie)라는 별칭으로 불린다. 한국으로 비교하자면, 아는 이들끼리만 낄낄거리며 치던 드립들이 페이스북, 유튜브, 인스타그램 등을 타고 번지면서, 소위 인싸개그가 되는 현상과 비슷하다고 보면 된다. 드립 중에서 최악으로 치는 드립은 설명해야 하는 것이다. 유머와 관련해서 자주 인용되는 말인 "유머를 설명하는 것만큼 재미없는 것도 드물다"는 말과도 통하는 면이 있다.

그래서 지상파 TV 방송, 정치권이나 정부 기관에서까지 해당 밈을 쓰게 되면 그 밈의 생명력은 끝장났다고 봐도 될 정도다. 예를 들면 '어쩔티비'가 인터넷에서 유명해졌다가 SNL 코리아, 미운 우리 새끼, 런닝맨[8] 등 여러 프로그램에서 언급되더니 사용하지 않게 된 것과 비슷하다. 인터넷 커뮤니티발 밈에는 해당 커뮤의 특수성이 있다고 보기 때문에 일반적으로 쓰이는 밈이 되는 경우 해당 인터넷 커뮤니티에서는 밈을 뺏겼다고 생각하여 재미없어서 사용하지 않는 경우가 있다.

하지만, 뉴스와 방송에서 만들어진 것이 밈이 되었다면 이례적으로 단시간에 기업과 예능에서도 사용하기도 한다. 인터넷 커뮤니티 파생으로 시작한 밈이야 커뮤니티, 유튜버, 일상생활 등을 거쳐서 어느 정도 보편적으로 퍼졌다고 판단할 때 방송국에서 쓰기에는 이런 경우에야 밈의 수명이 다했다는 말이 성립을 하지만, 뉴스를 타고 퍼진 밈은 파생된 것이 뉴스가 먼저이고 인터넷 커뮤니티는 나중에 확산되는 거라서 속도가 비교되지 않는 것이다.

대표적인 예시로는 내가 이러려고 대통령을 했나 밈이 있는데, 이 역시 유명인사의 발언으로 순식간에 전국민에게 각인이 되었고 그 즉시 방송국, 유튜브, 커뮤니티 가리지 않는 풍자 밈이 되었다. 전청조의 I am신뢰에요~도 뉴스에서 시작된 밈이 인터넷 커뮤니티에서도 사용하면서 유명해졌다. 사실, 대부분의 밈은 인터넷 커뮤니티에서 먼저 유행하고 TV를 비롯한 공중파에서는 나중에 나오는 경우가 많으니 유행에 뒤쳐진 것을 뒤늦게 따라하는 억지 밈처럼 보일 뿐이다.

밈을 리뷰하는 영상들에는 무조건 싫어요가 많이 박히며, 새 밈을 리뷰하고 대중화시키는 유튜버는 강한 증오를 받는다. 가장 많이 비난받은 사람은 Behind The Meme인데, 현재는 많은 비판 때문에 밈을 리뷰하는 것을 중지했고 2018년에 얼굴까지 공개한 뒤 사실상 채널을 동결시켰다.

한국과 일본에서는 그나마 오래된 밈을 재사용하는 경우가[9] 그럭저럭 있는 반면 해외에서는 오래되거나 이미 normie층에 유입된 밈을 철저하게 '죽은 밈(dead meme)'이라고 칭하며, 이미 죽은 밈을 사용하는 사람들을 normie로 간주한다. 이미 유튜브 등지에선 단순 번역을 통하여 해외에서 들여온 밈들이 호응을 얻으며, 그 출처를 밝히는 유저들에게 눈치까지 주는 분위기가 형성되었다.

이렇게 인터넷에서 밈의 영향력과 파장이 커지자 특히 욕설이 섞인 밈을 일반인에게 잘못 사용했다가 고소까지 당한 사례도 있으며, 밈 등의 인터넷 유행물의 원저작권자를 무시하고 자기들의 저작물이라고 저작권을 강탈하는 블랙 기업이나 인터넷 밈의 상표권을 멋대로 선제등록하고 자신의 밈의 권리자라고 우기는 개인이 등판한 사례도 있다.
2.1.2. 정치적 측면 관련[편집]
4chan, 9gag, 디시인사이드 등 우파 성향의 커뮤니티에서 만들어지고 향유되는 경우[10]가 많기 때문에 '우파적'이라는 인식이 있어서, 이로 인해 진보 진영에서는 밈과 밈 소비자들을 "밈적 사고(memetic thinking)'[11]에 찌들어서 사회를 오염시키고 있다"고 여기며 비판하는 경우가 많다. 이런 이중적인 행태를 비판하기 위해 만물일베설도 '밈적 사고'임을 언급하는 이들도 있으며, 한 트위터 유저가 김케장이 만든 '앗 아아' 밈이 노무현 전 대통령의 고인드립성 밈이라는 허위 사실을 유포하자, 케장이 2년만에 계정을 활성화하면서 반박하기도 하였다.

다만 상기 사이트들이 아무리 인터넷 커뮤니티를 주도한다고 한들 인터넷 커뮤니티에서만 밈을 만드는 것이 아니기 때문에 이러한 사고방식으로는 무한도전의 인스타그램에서 재조명된 무야호 같은 밈을 설명할 수 없다는 문제점이 있다. 실제로 수많은 밈의 근원지이자 좌파 성향을 가진 서브레딧이 많은 레딧도 있다. 게다가 밈에 정치색을 넣으면 대부분은 재미가 없어진다면서 사람들이 학을 뗄 정도로 싫어하는 경우가 많고, 심지어 최초에는 강한 정치색을 띠던 밈이 유명해지면서 정치색이 옅어지는 경우도 잦다.

한국에도 일베저장소를 비롯한 일부 극우 사이트들이 노무현 전 대통령에 대한 고인드립을 하는 등 정치밈을 만들어내는 경우가 많지만 이런 것들은 대부분 해당 커뮤니티 내에서만 소모되는 편이다. 그러나 홍준표 2번 같은 밈들은 정치찬, 보물창고, 이승빈 등의 방송에 나오며 유명해지자 사람들도 정치 성향에 관계없이 해당 밈들을 즐긴다. 문크 예거 관련 드립에서 정치색을 뺀 무지성도 많이 쓰이는 편이다. 이명박의 여러분 이거 다 거짓말인 거 아시죠는 정치인 관련 밈이지만 정치성이 옅어 자주 사용되는 밈이다. 한편 2022년 러시아의 우크라이나 침공을 계기로 밈 사용자들이 전쟁을 블랙 코미디로 소비하기도 했다.
3. 역사[편집]
세계최초의 밈
당신이 생각하는 당신의 모습
당신의 진짜 모습
인터넷 밈은 아니지만, 1920년 judge 잡지에서 인터넷 밈의 선조격 유머 그림이 실린 바 있다. With the college wit라는 대학교 관련 유머들을 모아놓은 페이지인데 두 가지 이상의 시각적 요소를 이용한 유머라는 점에서 세계 최초의 밈으로 여겨지기도 한다.#

현재까지 알려진 최초의 인터넷 밈은 1990년대의 미국 인터넷에서 이메일로 돌아다니던 동영상인 '베이비 차차'라는 아기가 춤추는 동영상이지만 이때는 '밈'이라고 말하지 않았다.[12]

2001년, MIT의 박사 과정 학생이던 페레티는 나이키 운동화에 '아동노동착취공장(sweatshop)'이라고 새겨줄 것을 주문했고, 나이키에서는 거절했다. 페레티는 이 내용을 친구들에게 보냈고, 이 이야기는 널리 퍼져, 마침내 NBC 투데이 쇼에서 나이키의 대변인과 토론까지 하게 됐다. 이것이 유전자 관련 용어였던 밈이 최초로 인터넷 용어로서의 '밈'으로 명명된 사례로 꼽을 수 있다. 페레티는 후에 “나는 리처드 도킨스가 밈이라고 부른 것을 우연히 만든 셈이죠”라고 썼다.

이처럼 영미권이 시초이기도 하고 오래 사용되며 자리 잡은 용어이나, 한국에서 단순 해외 신조어가 아니라 개념 자체가 실제로 대중에 소개된 것은 비교적 최근에 일어났다. 2020년 MBC 등 놀면 뭐하니?를 통해 비의 깡 유행이 소개되면서라고 볼 수 있다. 비의 깡이 재조명받은 과정이나 1일 1깡, 시무 20조 등의 파생 드립은 밈이라는 개념을 사용하지 않고서는 도저히 설명할 수 없는 것이었기에 당사자인 비 본인이 직접 밈이라는 단어를 사용하는 모습이 방송을 타기도 했다.

엄밀히 보면 밈 자체는 이미 한국에서도 존재하던 현상으로 기존에 자주 사용되던 '유행어', '농담', '드립', '합성 필수요소'라는 단어들이 이러한 개념을 지칭하였으나, 완전히 의미가 일치하는 것도 아님과 동시에 단어가 고작 한 음절에 불과하기 때문에 간편하다는 이유로 2010년대 후반 들어 앞선 단어와 용어들을 묶은 포괄적인 용어로 자주 쓰이게 되었다. 물론 이전에도 쓰는 경우가 있었으나, 상당히 마이너한 경향이 있었고, 실제 의미는 밈이 더 포괄적인데도 '합필' 하면 이미 한국 밈을 지칭하다 보니 '밈' 은 해외 밈을 가리키는 경향이 있기도 했다. 2020년대 들어서는 뉴스에서도 밈이라는 단어를 쓴다. 다만 아이러니하게도 대중매체를 통해 공식적으로 밈이라는 단어가 소개되는 것이 그 자체로 '밈'을 죽은 밈으로 만들 가능성이 있기 때문에, 밈이라는 개념이 한국 문화에 어떤 모습으로 자리 잡을지는 지켜볼 필요가 있다.

일본에서는 네타라는 말이 밈이라는 단어와 일맥상통한다. 나무위키의 전신인 리그베다 위키(엔하위키)에서도 네타라는 표현을 즐겨 썼기 때문에, 지금도 나무위키 문서 곳곳에서 네타라는 단어를 찾아볼 수 있다. 다만 네타라는 용어도 굳이 이야기하자면 합성 필수요소와 밈 개념의 중간에 가깝고 그 중에서도 합성 필수요소 쪽에 좀 더 가까운 편이긴 하다. 2020년대에는 일본에서도 와카 등 일시적인 파급력이 매우 크고 패러디를 양산해내는 성격의 네타를 넷 밈(ネットミーム)로 구분해 칭하는 경향이 있다.
4. 국가별 밈 목록[편집]
4.1. 대한민국[편집]
상세 내용 아이콘  자세한 내용은 밈(인터넷 용어)/대한민국 문서를 참고하십시오.
4.2. 해외[편집]
상세 내용 아이콘  자세한 내용은 밈(인터넷 용어)/해외 문서를 참고하십시오.
4.2.1. 일본[편집]
상세 내용 아이콘  자세한 내용은 밈(인터넷 용어)/해외/일본 문서를 참고하십시오.
5. 해외에서 파생된 밈[편집]
5.1. Geometry Dash 계열[편집]
Swag - 유명 유저 Riot가 Bloodbath 베리파이 영상에서 33% 부근에 'Swag'라는 소음을 내어 컬트적인 인기가 되었고, 이후 유저들은 Bloodbath를 클리어하는 영상에서 33% 부근에 'Swag'이라고 외치는 것이 유행이 되었다.
Kenos 베리파이 리엑션 - 문서 참조. Know Your Meme 링크
Have you send it to RobTop yet?
Back On Track
Michigun - Every level needs a triple(3단가시)
98% : Knobbelboy
소고기(Geometry Dash)[13]
2.2 when
2.2 Lobotomy
bro im already 5.3
GD Reference
Congregation jumpscare
fire in the hole
dash spider jumpscare
CBF detected, loser!
5.2. Grand Theft Auto 시리즈 계열[편집]
상세 내용 아이콘  자세한 내용은 Grand Theft Auto 시리즈/밈 문서를 참고하십시오.
All we had to do, was follow that damn train, CJ!
Big Smoke's Order
Grand Theft Auto IV의 로딩스크린
Ah shit, here we go again - GTA 산 안드레아스 시작부에서 CJ존슨이 하는 대사이다. 주로 무언가를 시작할 때 쓰이는 밈으로 2019년 5월에 가장 핫했던 밈 중 하나이다.
Lamar Roasts Franklin 2020년 초반에도 잠깐 흥하다 말았으나, 2020년 말에 사펑의 영향 때문인지 갑자기 부흥하게 된 밈이다.
You picked the wrong house, fool!
WASTED - 5편에서 플레이어가 사망할 때처럼 화면이 흑백으로 변하며 슬로우 모션과 함께 화면에 WASTED 이라고 뜨며 특유의 효과음이 나온다. 주로 뭔가 하려다 망하거나 고통스러운 장면 등에 합성으로 쓰인다.
5.3. 네모바지 스폰지밥 계열[편집]
상세 내용 아이콘  자세한 내용은 네모바지 스폰지밥/밈 문서를 참고하십시오.
5.4. 드래곤볼 계열[편집]
It's over 9000
Shoop Da whoop
브로리 MAD
I heard youre strong
5.5. 릭 앤 모티[편집]
Pickle rick - 위키피디아 know your meme
Evil morty theme - BGM이 자주 쓰인다.
Morty im bored im, im gonna kill you
5.6. 마다가스카의 펭귄 계열[편집]
Kowalski, analysis
I'm gonna say the N-word
5.7. 마리오 시리즈 계열[편집]
Hotel Mario
That’s mama luigi to you, Mario!
Thank you Mario! But our princess is in another castle.
Weegee
슈퍼크라운[14]
So long, gay Bowser!
Mario Pissing
5.8. 마인크래프트 계열[편집]
Creeper? Aw Man
Minecraft Crused Images
일부 마인크래프트 음악들 (Sweden, Minecraft 등)
Hey Shitass, wanna see me speedrun? - 유튜버인 드림과 관련된 밈.
히로빈
Minecraft Guy
Grotesque Steve
Hello young lady
You Died!
옛 피격 효과음
5.9. 별의 커비 시리즈 계열[편집]
Kirb
0% 0% 0%
구르메 레이스
별의 커비(애니메이션)
I need a monster to clobber that there kirby.
Surely you Jestin.
He don't scare me none!
Kirby Explains...
Kirby Falls
This upset kirby immensely
식칼 커비
I Am No Borb
한 커비 팬이 그린 만화이다. 만화의 내용은 디디디 대왕이 커비에게 욕설을 가르치는 내용인데,[15] 어느 장면에서 메타 나이트를 보고 디디디가 "Bad Orb"라고 하자 커비가 Borb라고 말해 컬트적인 웃긴 밈이 되었다. 이후 이 밈은 커비 말고 다른 매체의 팬들에게서도 패러디되고, 메타 나이트에게 Borb라고 별명을 붙이는 사람들이 늘어났다.
* Kirby calling the police
5.10. 소닉 더 헤지혹 시리즈 계열[편집]
It's no use!
PINGAS
Sanic hegehog
That tornado is carrying a car!
Uganda Knuckles
You're too slow!
Eggman’s announcement
You’re too late Sonic! I’m forklift certified!
5.11. 팩맨 시리즈 계열[편집]
Here Comes Pacman
Pac is Back
Nerd Pac man
팩맨 월드 3의 대사들
Details of my sector's energy shouid be between me and Ms. Pac, thank you very much.
헬로 팩맨!(팩맨 2: 새로운 모험)
5.12. 슈렉 시리즈 계열[편집]
oh hello there
Shrek is love, Shrek is life
What are you doing in my swamp
5.13. 스타워즈 계열[편집]
스타워즈 오리지널 트릴로지
Han Shot First
I AM YOUR FATHER
It's a trap!
스타워즈 프리퀄 시리즈
오비완 케노비
Flying is for droids
Sith Lords are our speciality
Hello there!
So uncivilized
Only a Sith deals in absolutes
I HAVE THE HIGH GROUND!
다스 시디어스
DO IT[16]
Did you ever hear the tragedy of Darth Plagueis the Wise?
Ironic
I love democracy, I love the republic
I AM THE SENATE / It's treason, then
UNLIMITED POWERRRRR!
Execute Order 66
아나킨 스카이워커
I HATE SAND
This is where the fun begins!
This is outrageous, it's unfair
What have I done?
Because of Obi-Wan?
I HATE YOU
NOOOOOOOOOOOOOO
파드메 아미달라
For the Better, Right?
그리버스
A fine addition to my collection
로그 원: 스타워즈 스토리
오슨 크레닉
We were on the verge of greatness, we were THIS close
Oh, it's beautiful
WE STAND HERE AMIDST MY ACHIEVEMENT, NOT YOURS!
Are we blind? Deploy the garrison!
K-2SO
Congratulations! You are being rescued
치루트 임웨
I am one with The Force and The Force is with me
반란 연합 장교[17]
We're detecting a massive object emerging from hyperspace
기타
Good soldiers follow orders
I don't like it. I don't agree with it. But I accept it
I have spoken
THIS IS THE WAY
What about the droid attack on the wookiees?
Take a seat, Young Skywalker.
5.14. 심슨 가족 계열[편집]
Marge Krumping - 원본 심슨 가족 시즌 19 에피소드 6화에서 마지 심슨이 춤추는 영상인데, 이상하게 영상 전체가 아닌 이 부분만 따와서 여러 가지에 합성을 한다. Know your meme 설명
Moe's Dancing - 모 시즐랙 문서의 여담 문단에 적힌 것이 아닌 이것이 밈이다. 대충 이런 식으로 합성된다. 참고로 더블배럴 샷건에서 세 발이 발사된다.
Homer Simpson Hide In Shrubs - 호머 심슨이 나무 울타리로 숨는 장면. 쥐구멍에라도 숨고 싶은 심정을 나타낼 때 사용한다.
바트의 죽음
Sweet Merciful Crap - 심슨에 나오는 대사 중 하나. 심슨이 엉망진창이 된 자기 차를 보고 경악한 장면이다. 대충 해석하면 '(아이고) 이럴 수가!!!' 정도가 된다. 또한 SMG4가 이 대사를 자주 사용한다.
Steamed Hams
You got the dud - 호머 심슨이 밀하우스 카드를 보고 웃는 장면을 밈화시켰다. 2017년 후반기에 인기 있었던 밈.
Bart hits homer with a chair
My Kia - 바트 심슨이 학교 앞 나무 밑동을 폭발물로 날려보냈는데, 그 나무 밑동이 교장의 차량인 흰색의 기아 프라이드 차량을 박살 내버리자, "My Kia!" 하면서 통곡하는 것이 압권.
충공깽
무슨 마약하시길래 이런 생각을 했어요?
5.15. 유튜브 리와인드 2018[편집]
That's Hot - 2018년도 영상에 출현한 윌 스미스가 영상의 막바지에서 망원경으로 포트나이트 버스를 보면서, "Ah, that's hot! that's hot…"이라고 말한 것인데, 망원경으로 보는 것을 다르게 만들어 웃음을 자아내는 밈이다. 예시
I am so proud of this community
5.16. 이니셜 D 계열[편집]
유로비트 - Running In The 90's, Deja Vu, Gas Gas Gas 이 세 유로비트곡이 대표적으로 손꼽힌다. 이니셜 D의 밈은 자동차나 사물들을 이용해 드리프트를 묘사하는 영상에서 갑자기 뜬금없이 유로비트 곡이 나온다.[18][19] 경우에 따라 영상 위에 [Eurobeat Insenfies](격렬한 유로비트)를 써넣기도 하며 그 외에 이니셜 D 애니메이션, 게임에서 등장한 다른 유로비트 곡도 사용된다.
관성 드리프트 - 애니메이션 기준 1기 1화에서 타카하시 케이스케가 관성 드리프트를 하는 장면을 보고 놀라는 장면이다. 유로비트와 함께 이니셜 D를 대표하는 밈.
이로하자카 점프 - 애니메이션 3기에서 코가시와 카이가 이로하자카 특유의 고저차를 이용해서 점프를 하는 장면이다. 이때 사용되는 유로비트 곡은 이니셜D 3기에서 코가시와가 고저차를 이용해서 점프할 때 나오는 Crazy For Love. 패러디가 되면 게임으로 패러디를 하는 경우가 대부분이며, 실사에서는 차량이 점프를 하다가 사고가 나거나, 추락 사고를 당할 때 나오는 경우가 많다.
5.17. 포켓몬스터 계열[편집]
Surprised Pikachu 사진 예시
Who's That Pokémon? - 직역하면 "이건 어떤 포켓몬일까요?" 로 해석 가능하고, 한글판으로 따지면 "오늘의 포켓몬은 뭘까요?" 정도로 보면 된다. 보통 딱 봐도 알 것 같은 포켓몬일 것 같은 정답이 아닌 기출 변형 내지 파괴 수준으로 전혀 다른 녀석이 정답으로 나온다.
It's super effective! - "효과가 굉장했다!"
so i herd u liek mudkipz - 물짱이 관련 밈이다.
POKÉDANCE
5.18. 레인보우 식스 시즈 계열[편집]
로드 타찬카 - 레인보우 식스 시즈의 오퍼레이터 타찬카가 사용하는 가젯에 비해 낮은 유용도를 보이자 아예 유저들이 타찬카를 밈화시키기 시작한다. 개발사인 유비소프트에서도 이 밈을 아주 잘 인지하고 있는지라 타찬카의 리워크 당시 아주 약을 빤 영상을 제작했다.
퓨징하기 - 위에 서술한 타찬카와 같은 스페츠나츠[20]대원인 퓨즈의 가젯[21]이 아군, 적, 인질 가리지 않고 잡는 바람에 유저들이 이 역시 밈으로 만들었다. 이 역시 유비소프트가 잘 알고 있어서 이후에 출시한 퓨즈의 정예스킨에 인질을 사살하란 문구를 숨겨놓았었다.
와마이 댄스 - Y4S4에 추가된 와마이가 자신의 가젯인 자석을 들고 qeqe 하는 게 무언가 멍청해 보여 생겨난 밈. 원래 다른 가젯을 가진 오퍼에이터로도 일부 존재하였으나 와마이 특유의 표창을 쥐는 듯하는 모습 때문에 생겨났다.
5.19. 팀 포트리스 2 계열[편집]
상세 내용 아이콘  자세한 내용은 팀 포트리스 2/밈 문서를 참고하십시오.
외국 밈의 필수요소.
6. 시기별 유행 밈[편집]
대한민국에서 유행했었던 한국발 밈, 외국발 밈의 모든 것과, 인터넷상이나 현실에서 유행했었던 사회적, 문화적 현상 또는 유행어를 월간별로 기술한다. 각각 년도마다 월간별 기술 조건은, 그 현상이 오래전부터 쓰이기 시작했지만 대중적인 인지도가 없이 특정 커뮤니티 내부에서만 쓰였을 때가 아닌, 대중적으로 유행하기 시작한 년도를 기술한다. 그리고 특정 커뮤니티 내에서만 유행하다가 대중적으로 퍼지기 전에 사그라든 현상은 기술하지 않는다.
6.1. 대한민국[편집]
한시적 넘겨주기 아이콘  서술되지 못한 밈이 많으며 더 많고 자세한 한국 밈을 알고 싶다면에 대한 내용은 밈(인터넷 용어)/대한민국 문서를 참고하십시오.
2000년대에는 인터넷이 생기면서 밈과 유행어들이 많이 발생했고 2010년대에 들어서는 스마트폰의 보급과 인터넷의 발달로 SNS, 커뮤니티 등지 등 인터넷에서 발생한 밈들이 많아졌다.
6.1.1. 1997년, 1999년[편집]
1997년
8월: 무대뽀, 헝그리 정신
1999년
시기 불명: 디지몬 어드벤처, 볼레로
10월: 난 한 놈만 패
6.1.2. 2000년대[편집]
2001년
3월: 느그 아부지 뭐하시노
7월: 성지순례
9월: 내가 제일 존경하는 오사마 빈 라덴[22]
10월: 초성어[23]
2002년
2월: 안톤 오노
6월: 붉은악마, 압박, 오 필승 코리아, 옥동자
7월: 개죽이
8월: 긴또깡
9월: 짤방[24]
10월: 내 아를 낳아도
12월: 불심으로 대동단결
2003년
1~4월: 맞습니다 맞고요, 뷁, 밥은 먹고 다니냐
8~9월: 두 번 죽이는 거, 숫자송
11~12월: 어 작업중이야[25], 을용타
2004년
6~12월: 이 안에 너 있다, 파맛 첵스, 적절하다
2005년
4~9월: 술은 마셨지만 음주운전은 하지 않았다, 신돈, 허이짜
2006년
2월: 미녀는 석류를 좋아해
12월: 바이오하자드 4 Mindless Self Indulgence Shue Me Up 매드 무비
2007년
1~2월: 망했어요
3월: 나는 관대하다
6월: 아시발꿈
7월: 빵상 아줌마
8월: 여러분 이거 다 거짓말인 거 아시죠
9월: 블리치 소울 소사이어티 AMV
10월: 데스노트/애니메이션, 카르미나 부라나 중 O Fortuna, 바카야로이드, 도박묵시록 카이지/애니메이션(인간 관악기 MAD 포함)
2008년
2월: 야 기분 좋다, 현기증 난단 말이에요
7월: 어둠의 다크에서 죽음의 데스를 느끼며, 빠삐놈
8월: 까방권, 두뇌풀가동
10월: ONE OUTS/애니메이션
2009년
9월: 꿀벅지
10월: 컨트리볼, 폴란드볼, 인실좆
11월: 루저의 난, 빵꾸똥꾸, 빵셔틀
12월: 한 뚝배기 하실래예[26]
6.1.3. 2010년[편집]
1월: 넌 나에게 모욕감을 줬어[27], 돋네, 박대기, 오덕페이트, 코렁탕
3월: 삼일절 사이버 전쟁[28]
6월: 아모캣[29]
7월: 에바, 함정카드 밈, 꼽등이
8월: 아직 한 발 남았다[30], 힙통령
9~10월: 꽈찌쭈, 아니 없어요 그냥
12월: 종범
6.1.4. 2011년[편집]
1월: 간 때문이야, 씹선비
4월: 너 고소, Nyan Cat
5월: 김치녀
6월: 내가 제일 잘 나가
7월: 개 짖는 소리 좀 안 나게 하라
8월: 11미터 모형탑 훈련
9월: 더 이상의 자세한 설명은 생략한다., 이 차는 이제 제 겁니다
11월: 이봐요 미친놈씨
12월: 도지삽니다
6.1.5. 2012년[편집]
1월: MC무현, 뽐거지
2월: 느그 서장 남천동 살제?, 작은 하마 이야기, 저도 참 좋아하는데요, 제가 한번 먹어보겠습니다
4월: 네가 하면 나도 한다
7월: 강남스타일
8월: 북경 정씨
9월: 무대를 뒤집어 놓으셨다
10월: 어서 와, 인증 대란, 자박꼼
6.1.6. 2015년[편집]
1월: sake L
3월: 오로나민C, Yee
4월: 눈을 왜 그렇게 떠?
5월: 마이 리틀 텔레비전
6월: 누가 기침 소리를 내었는가
7월: 나 꿈꿨어 귀신 꿈꿨어
9월: 어이가 없네
10월: 레스토랑스
6.1.7. 2016년[편집]
2월: 아기상어
4월: 샤샤샤
5월: 뭣이 중헌디
6월: 너 때문에 흥이 다 깨져버렸으니까 책임져
8월: PPAP
10월: 너굴맨, 너무해 너무해
11월: 내가 이러려고 대통령을 했나
6.1.8. 2017년[편집]
8월: 산타벌스
9월: 트로피카나 스파클링
10월: 항아리 게임
11월: 상상도 못한 정체
6.1.9. 2018년[편집]
1월: 사랑을 했다, 폰은정
2월: 였던 것 드립, 영미
3월: 병신TV
4월: 탈모르파티, 어벤져스: 인피니티 워
5월: 다 아는 사람들이구먼, 조혜련의 태보 다이어트
6월: 나는 예쁘지 않습니다
7월: 오또케, He is ○○
8월: 착짱죽짱
9월: 조혜련과 태보의 저주
10월: 야 XX, 껍질 미리 깐 달걀
11월: 소년점프
12월: 핵인싸
6.1.10. 2019년[편집]
1월: 절대태보해
2월: 치카 댄스, 대충 드립, 자전차왕 엄복동
3월: 안녕하살법
4월: 버억, 힘들지 않아 거친 정글 속에 뛰어든 건 나니까 암오케
5월: 나도 내가 징징거리고 눈꼴시려운건 알고 있는데, 쁘악수
6월: 기생충, Woman Yelling at a Cat, 틀니 2주 압수
7월: 2019년 일본 상품 불매운동, 날강두, 이 시국에, 인사하는 제리
8월: 이이잉 앗살라말라이쿰, 곽철용, 유튜브 알고리즘, 여름이었다, 잼민이
9월: 오뚝이 다트, 찌리찌리, 헬창, 국밥충, 어벤져스: 엔드게임
10월: 가성비 댓글, 코리안조커
11월: 펭수, 아마존 익스프레스, 겨울왕국 2
12월: 단또단또, 야나두
6.1.11. 2020년[31][편집]
1월: 아임뚜렛, 안전가족[32], 그랜절, 지이잉, 호박고구마, 할카스
2월: 아무노래, 코로나바이러스감염증-19
3월: 종로로 갈까요, 만희물, ㄷㄷㄷㅈ, 나비보벳따우, 죄송합니다, 미안하다 이거 보여주려고 어그로끌었다
4월: 관짝춤, 나는 개인이오, 캬루
5월: 깡, 헬테이커, 펀쿨섹좌, 엄준식
6월: ~은(는) 사드세요.....제발, 불 좀 꺼줄래?, 야 꿀벌, 가면라이더 드립
7월: 첵스 파맛, 군침이 싹 도노, 가짜사나이, 부리부리 마신 소환 춤, ~를 알려주겠다
8월: 뒷광고, 테스형!, 애기공룡 둘리, 돌리랑 도트가 제일 좋아, 사실은 오래전부터 당신 같은 XXX를 기다려 왔다우
9월: 제주도 찐 사투리, 어몽어스, 빅맥송, 다메다메, 펫 더 피포
10월: 훈발놈, Team Azimkiya, 뭉탱이, 난 너가 줏대있게 인생 살았으면 좋겠어
11월: 치즈분수, 한국인이 좋아하는 속도
12월: 사쿠란보, 하츠네 형돈, Pop Cat, 당연히 말이 되죠
6.1.12. 2021년[편집]
1월: 무야호, 팝캣, 강아지 경찰 아저씨, 포브스 드립, 스윗중남, 비트 밈
2월: 롤린, 삼각함수송, 제 딸을 살해한 놈들을 15년 후에 죽여주세요, 무한열차, 안아줘요
3월: 가수 일론 머스크, 시무라 아주머니, LH 게이트
4월: 멈춰![33], 마이야히, 오타쿠가 정치글을 쓰게 만드는 정부가 존재할 필요가 있나?, 육군 we 육군
5월: 머니게임, 나락송, 아라아라, 흔한 여고생들의 뻘짓, 프라이데이 나이트 펑킨, 삼성걸, 근데 이제 뭐함?, 넥스트 레벨, 굿바이 선언
6월: 꽁기깅깅깅공강강꽁기깅깅꽁기깅강, 제로투 댄스, 한심좌, 샤방송
7월: 똥 밟았네, 누가 칼들고 협박함?, 마인크래프트 19금 사태, 2020 도쿄 올림픽 픽토그램
8월: 활벤져스, 소금소금소금팥팥팥, 잭 오, 그런건 없다 게이야, 성윤모
9월: 오징어 게임, 앙카 존 댄스, 라고 할 뻔
10월: 스트릿 우먼 파이터, 설거지론, 커다란 자갈돌은 짱돌, 파피 플레이타임
11월: 오니기리 댄스, 어쩔티비, 정상수, 몰?루, 개같이 XX, 북방코끼리바다표범
12월: 여자가⋯ 말대꾸⁈, 요나요나 댄스, 어느새부터 힙합은 안 멋져, 힘숨찐, 슈퍼 아이돌, 사실이 아닙니다!, 예쁘잖아
6.1.13. 2022년[34][편집]
1월: 피에로 드립, 미스터 인크레더블 밈, 코카인 댄스
2월: 곽윤기 뒷선수 시점, 저 반바지 아니에요
3월: 포켓몬빵, 킹근육골드몬, 지구방위대 챌린지, 어몽어스 잼민이
4월: 부럽지가 않어, 해병문학, 아냐 포저 얼굴 표정, 마이 네임 이즈 찢기, LOVE DIVE
5월: 마미 롱 레그, 소울리스좌, That That, 아이 러브 마이 화자 마자 브라자, 눗눗, 범죄도시 2
6월: 재즈를 뭐라고 생각하세요?, 후드에서 살아남기, 여기어때송, 오붕가
7월: 이상한 변호사 우영우, 닭팔이, 모기송, 점점 빨라지는 노래
8월: 제네시스좌, 슬픈 고양이, 인주 앨리스, 알빠노
9월: 실패작 소녀, Young한데? 완전 MZ인데요?, 새삥, 제로투 회피
10월: 중요한 것은 꺾이지 않는 마음[35], 갱갱갱
11월: 띠용흥민, 알빠임?, 귀여워서 미안해, 짱구야 아빠를 속인거니?
12월: 오빠 오빠 차 있어?, 뉴진스의 하입보이요, 근데 어쩔건데?, 폼 미쳤다
6.1.14. 2023년[36][편집]
1월: 닛몰캐쉬, 토카토카 댄스, 04년생 클럽춤. 팅팅송, 멋지다 연진아, 스튜어디스 혜정아, 제기랄, 또 ~~~야!, 뉴진스의 하입보이요, 우리 엄마엄마가
2월: 시그마, 마따끄, 나문희의 첫사랑, 나이트 댄서
3월: 고도로 발달한 A는 B와 구분할 수 없다, 스즈메의 문단속, 인터넷 야메로, 뽀삐뽀삐뽀 뽀삐뽀
4월: 크리스천, 봇치 밈, 해피캣, 이궈궈던
5월: 아디아디아디, 강풍 올백, ~한 건에 대하여, 【최애의 아이】, 와플, 삐에용 부트 댄스
6월: 너 T야?, AI 커버, 사카밤바스피스, 스키비디 토일렛, 사우스 코리안 파크, 그리메이스 셰이크, 퀸카
7월: 크아아악! XX아!, 지구 온라인, Gucci Boy, 아오 페리시치, 좋았어 영차[37], 엘리멘탈, Steal The Show
8월: LK-99(초전도체), 인생 하드모드, 모스부호, 어 맞아맞아 놀랍지만 그건 사실이야, 피어나 너 내 도도독
9월: 올리버쌤 성대모사, 탕후루, 워싱시, Love Lee, Miku
10월: 로리신 레퀴엠, 김도와 마즈피플, 슬릭백, 잘 가라, 최강. 내가 없는 시대에 태어났을 뿐인 범부여, I am신뢰에요~, 어머니가 계셨구나, 똑똑한 청년.
11월: 장충동왕족발보쌈, 이비온 악기, 세 세구 라이드, I am신뢰에요~, 럼자오자레, 88848, 당근칼, 기습숭배
12월: 나루토 춤[38], 나 XX인데 개추 눌렀다, 조시 허처슨, (XXX에서 연락옴), 치피치피 차파차파, 전두광[39]
6.1.15. 2024년[편집]
1월: 치피치피 차파차파, 꽁꽁 얼어붙은 한강 위로 고양이가 걸어다닙니다, 춤추는 늑대, 탕후루는 차갑다, 은행 플러팅, 투슬리스 댄스, 국가권력급, 밥똥던, 한잔해, 움파룸파춤
2월: 캣냅, 브링방방, 손흥민 축신짤, 움파룸파춤, 첫 만남은 계획대로 되지 않아, 목욕송
3월: 밤양갱, AK47 맞고 사망한 외할머니, 구구단 댄스, 래빗 홀, 정몽규 나가, 원영적 사고, 솔랭의 제왕
4월: 야레야레 못말리는 아가씨, 무뼈궁무닭발, 포켓몬 댄스, 프리터 댄스, 꽁꽁 얼어붙은 한강 위로[40], 난 대학시절 묵찌빠를 전공했단 사실, HAPPY, 크아악 롤랑 이 새끼가, 어라운드 더 월드
5월: Pedro Pedro Pedro, ㅎㅎ즐거우세요?, 맞다이로 들어와, 완전 럭키비키잖아, 평화누리특별자치도, 직구 금지, 점심을 든든히 먹어두어라 저녁은 지옥에서 먹을테니, 마라탕후루, 포철고 챌린지, 경북대학교 에브리타임, 귀여워ㅋㅋ 키스할래요?, 정상화(팩트는 ~라는 거임), 수수수수퍼노바, 푸른 산호초
6월 : 형이랑 내기할래?, 류정란 챌린지, 사랑했나봐[41], 츠~긍슴승 승르흐르르~, 시카노코노코노코코시탄탄, 버튜버 페이셜 테스트, 티라미수 케익, 하루만 기다리면 ○○가 나와요!, 뉴진스럽다, 누울래?[42], 오물 풍선, English or Spanish, 폴라레티, 메스머라이저
7월 : 네 기꺼이, 외모 췤!, 아로나 댄스, 줄게, 신창섭, 쿵쿵따, 미룬이 사건, 삐끼삐끼 춤, 인사이드 아웃 2, 김쁠뿡, 조이는 보이
8월 : 창팝(다 해줬잖아, 바로 리부트 정상화, 이해가쏙쏙되잖아리슝좍아), 삐끼삐끼 춤, 천문학자 밈, 2024 파리 올림픽/양궁, 유빈양 할머니, 김쁠뿡, 코난 춤, 대사나열연습
9월 : 퍽 아디다스, 콘코드, 개고기 미트볼, 두 수 앞, 두들 댄스, 조이는 보이, 티니핑, 바니토사, 흑백요리사: 요리 계급 전쟁 시즌 1(최강록, 유비빔, 고죠 백종원), 마이타케, 제로 산소, 마루는 강쥐
10월 : 마이타케, 두들 댄스, 흑백요리사: 요리 계급 전쟁 시즌 1(검은 장갑, 영역 전개), 옴브리뉴, 자낳괴 탄생비화, 반전 어둠 챌린지, 조커: 폴리 아 되, 버니스 댄스, 프로토 디스코, 피크민
11월 : 아파트, 터미널 댄스(실루엣 챌린지), Faces, ○덕여대, 차가운 상어 아가씨, Queen never cry, 까탈레나, 샹하이 로맨스, 시그마 보이
12월 : 계엄, 봇이지 뭐, 샹하이 로맨스, 바람피면 D지는거야, 미무카와 나이스 트라이, 테토리스, 오징어 게임 2(얼음!!, 저는 이 게임을 해봤어요!, 둥글게 둥글게, 타노스 랩), 안주거리, BRAIN, Không Sao Cả(괜찮아 챌린지)
6.1.16. 2025년[편집]
1월: Chill guy, 내면의 기가차드, 나쁜 답변 좋은 답변, 오징어 게임 2(성기훈 - 얼음)
2월:
3월:
4월:
5월:
6월:
7월:
8월:
9월:
10월:
11월:
12월:
6.2. 해외[편집]
상세 내용 아이콘  자세한 내용은 밈(인터넷 용어)/해외 문서를 참고하십시오.
레딧의 r/memes와 r/dankmemes를 참고.
LIMC, Layzzz 밈의 유래를 설명하는 채널로 사회현상을 탐구하듯 진지한것이 특징이다, 업데이트가 매우 빠르다. 비슷한 웹사이트로 Know Your Meme이 있다. 정확히 언제 유행했는지 궁금하면 밈 제목/이름 Google Trends도 꽤 괜찮은데 지역을 전 세계로 설정하는 것이 더 결과가 정확하다.
6.2.1. 1996년[편집]
8~12월: Dancing Baby (Baby Cha-Cha)[43]
6.2.2. 2000년대[편집]
2007년
5월: 릭롤링
7월: 미트스핀
9월: 2 Girls 1 Cup
10월: 케이크(포탈)
2008년
6월: Loss
2009년
7월: fsjal
6.2.3. 2012년[편집]
2월: me gusta, rage guy
7월: 강남스타일
6.2.4. 2013년[편집]
7월: ayy lmao
6.2.5. 2016년[편집]
1월: Drakeposting, Ain't Got Rhythm
5월: Dat boi
6월: Arthur's fist
12월: TheLegend27
6.2.6. 2017년[편집]
1월: Brother, may I have some oats?, Salt Bae
2월: Expanding Brain
3월: Trash Doves, 🅱️
4월: Thank you Kanye, Very cool!, Fidget Spinner
6월: #thefloorislavachallenge
7월: Crash Bandicoot Woah, Guess I'll Die
8월: Excuse me what the fuck, Distracted Boyfriend
10월: S Stands for...[44]
6.2.7. 2018년[편집]
1월: Ugandan Kunckles, Tide Pods, Somebody Toucha My Spaghet, Burger King Foot Lettuce, BitConnect
2월: Steamed Hams, Drake, Where's Doorhole?, Change My Mind
3월: Gru's Plan, Peter Hurts His Knees, Globglogabgalab
4월: Kird, Walmart Yodel Boy, I don't feel so good
5월: Markiplier E, This Is America, Bart Hits Homer With A Chair, Yanny Or Laurel, Is this a pigeon?
6월: EVERYONE IS HERE
7월: ligma
8월: Excuse me what the fuck
9월: Bongo Cat, THANOS CAR, Moth & Lamp[45], First Time?
10월: Kowalski, Analysis, Unexpected Sans
11월: Surgery On a grape, Tik Tok (I'm Already Tracer, Hit Or Miss), Day XX of No Nut November[46]
12월: Wanna Sprite Cranberry, Ahh Thats Hot
6.2.8. 2019년[편집]
1월: Sicko Mode or Mo Bamba, Big Chungus, What the Fuck Did You Just Bring Upon This Cursed Land, Thats How Mafia Works, Miles Morales Flirting
2월: Let me in!, Shaggy, Will Smith's Genie
3월: Ricardo Milos, Florida man, Verbalase Tetris Beatbox, You was at the club
4월: 듀오링고 부엉이[47], Ah shit, here we go again, Tuxedo Pooh, We have food at home, Unsettled Tom
5월: Fortnite bad, Minecraft good, Sonic, Oh Shit A Rat
6월: Toothless Presents Himself, Stonks, Keanu Reeves, They're not loot boxes, they're surprise mechanics, Me and the Boys, WikiHow
7월: 마인크래프트 (CREEPER? AW MAN!), JSA에 서있는 김정은과 트럼프, Awesome (Scatman's World), Gamer Girl Bath Water[48], Storm Area 51, Say Sike Right Now, CEO of racism, Not funny, didn't laugh, Yeah, this is big brain time.
8월: Ah yes, enslaved (meme)., Video games cause violence, Beanos, Aight, Imma head out
9월: September, Storm Area 51, 69, Mike Wazowski-Sulley Face Swap, change da world my final message. Goodb ye
10월: Vibe Check, average fan vs average enjoyer
11월: Æügh
6.2.9. 2020년[편집]
1월: World War III, Dorime, Penis music, Meme Man, Financial Support, 퓨디파이의 안식월
2월: Sad Linus, 레고 시티, 코로나바이러스, mmm monkey
3월: Dog Smothering Owner[49], Npesta Kenos Reaction
4월: Coffin Dance, The Man Behind the Slaughter, Knuckles Meme Approved, Hey, wanna listen to some tunes
5월: Mundial Ronaldinho Soccer 64
6월: Femboy Hooters, Wide Putin Walking, Xue Hua Piao Piao, My Superhero Movie
7월: Baka Mitai / Dame Da Ne[50][51], Get stick bugged lol, Wait, it's all Ohio? (Always has been)[52], Are ya winning, son?, car shearer dance
8월: BOBUX, American Cup Song, Distraction Dance, But If You Close Your Eyes
9월: Among Us[53]
10월: Da Vinci, Limmy Waking Up
11월: Indihome Paket Phoenix[54], PET THE PEEPO, Zero Two Dance, Exaggerated Swagger of a Black Teen, My Little PogChamp, Let's Go That's Class
12월: Pop Cat, Cyberpunk 2077[55], Lamar Roasts Franklin, Drip Goku
6.2.10. 2021년[편집]
1월: drip goku, GrubHub[56], Bits, 게임스탑 주가 폭등 사건[57]
2월: Morshu RTX, When the Imposter is sus!, oh no, Refuses to elaborate further
3월: mmmm cow[58], 마이야히, DaBaby
4월: I used to roll the dice, 아라아라
5월: youtube kids be like[59], Think, Mark! THINK!!, Drip Car
6월: SAM, I forgor 💀, Gigachad
7월: I have family, Lore
8월: 잭 오 챌린지
9월: 오징어 게임, Steve you gotta help me I'm stuck, god, please[60], Ankha Zone[61]
10월: Shang Abi (Dancing Asian Man), Snotty boy glow up
11월: 오니기리 댄스, Super idol, Damn Daniel ar ar ar, You should kill yourself Now[62]
12월: YONA YONA DANCE, Christmas is next Friday /
Christmas, just a week away![63], Mr. Incredible Becoming Uncanny
6.2.11. 2022년[편집]
1월: Mr. Incredible Becoming Uncanny, cupcakke, Standing Here I Realize, Saul Goodman 3D[64], Oh Boy, My Favourite Seat!, Hatsune Miku does NOT talk to british people
2월: No Bitches?, Pushin P, Talking Ben The Dog, Fortnite Battle Pass, iShowSpeed Talking Ben
3월: There will be bloodshed[65], Skeleton Roasting JellyBean, 77 + 33 = 100, Misery x CPR x Reese's Puffs, Jamie Big Sorrel Horse
4월: Will Smith Slapping Chris Rock, Guys look a birdie, Quandale dingle (goofy ahh), Anya Forger Face
5월: The Backrooms, Terrified Noot Noot, It's Morbin Time, My Dog Stepped On a Bee, turi ip ip ip[66]
6월: Blue Lobster Jumpscare[67], My reaction to that information[68], Nextbot, Exposed Nerve, Squidward kicks of his house, Sad Cat Dance[69]
7월: Sims cat break-dancing, Demon core, YOU IS A, Kratos Falling, WHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT
8월: WHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAT, Nanomachines, son!, Top 5 series, Spinning rat, VS 놀이[70], Dr. Livesey phonk walk[71], California Gurls
9월: Trollge, Vergil Status[72], Zyzz[73], One Piece Is Real[74], Slander, Cooking videos[75], Gigachad
10월: Gigachad, Mutsuki Dance, Alphabet Lore[76], Wise Mystical Tree, 사이버펑크: 엣지러너[77], Dream face revealed, Baller
11월: Only in Ohio[78], No Nut November, Mississippi Queen[79], Satsuki ask, Zero Two Dodging, Sinner's hand[80], Ghost stare
12월: Koopa dance, The Lightskin Stare, Mario Movie Trailer, chip[81], SIU, Phonk Epic Battle, Choo-Choo Charles, xQc goes from happy to sad
6.2.12. 2023년[편집]
1월: Sigma, KON, Skibidi Bob Yes Yes[82], Waffle House chair[83], Wednesday[84], Maxwell cat[85], Let me do it for you[86], The boys, Boykisser[87]
2월: da biggest bird[88], Death Meme, Oklahoma, Toca Toca, Noor, Excuse me brah, Listening to ~ be like[89], Whopper Whopper, Metal Pipe Falling, Animan Studios[90], Maxwell the cat
3월: Lario, Cammy Stretch, Waffle falling over, Biden & Trump Plays[91], Bruce Wang, Atomic Heart, Singing Killed My Grandma, Opila Bird, unstoppable, Makeba, Fake Mr. Beast
4월: Jedi Killer, I'm Only Human, Wolf jumpscare, John Pork, Balenciaga AI Video, Serbian dancing lady, goku prowler, Bronze age shitpost, My Heart Is Cold, Buckle, Give me candy, Aidoru, Mike O'hearn, Peaches
5월: Pizza Tower Screaming, Minecraft Dirt Block Guy, woah pipe bomb, Which one is not sleeping, I am a surgeon, happy cat, Waffled, Adam Warlock flying, Babies Do Idol Dance, This is a, Brawl Stars Rank Up, Banana Cat
6월: Monday Left Me Broken, We're in Heaven, Ambatukam, Pride Month Demon, Frigigigi, Leave me alone, Blud, Cameraman:, Bagel Effect, ai sponge, Shadow wizard money gang, If I roll a, Sacabambaspis, One Two Buckle My Shoes, John Cena Dancing with Headphones
7월: Mephisto, Elephant Mario, Poland Is Everywhere, Nuh Uh, The Grimace Shake, Dog Sasuke, Cupid, Barpenheimer, Ukulele Apology, 페니 파커, Threads, 머스크 vs. 저커버그, Shuffle, Shizzy, Skibidi Toilet, Rainbow Friend, Steal the Show, Elemental
8월: Police I swear to god, One must imagine Sisyphus happy, LK-99, On Sight Bear, Vtuber Concert meme, Pharao's curse, Red Flags, Windows Punch, Is that gabe, More Passion, More Energy, Freddy Fazbear
9월: Color, IShowMeat, Donald Trump's Mugshot, Stock Shuffle, Hakari Dancing, Smurf Cat, Strawberry Elephant, Supïdo, 디지털 서커스, We live we love we lie, Miku
10월: Kagura Bachi, Polar bear 2026, 95% of NFTs Are Worthless, 숙성 로리신레쿠이에무[92], Squint Your Eyes, Opium Bird[93], Sticking out your gyatt[94], Jubi Slide, Gambare Gambare Senpai, Slickback, All My Fellas, Uhu Cat
11월: Huh cat, All My Fellas, The Coffin of Andy and Leyley, jetcar24.ru, Belligol[95], 9mm go bang, Freddy Fazbear[96], Chica rizz, Napoleon
12월: 조시 허처슨, Sam Sulek, Nah I'd Win, The Frog Video, Dream VS Gumball, 후드 노숙자 밈, Kawasaki, Gotta Lock in, I Think We're Gonna Have To Kill This Guy Steven, Chipi Chipi Chapa Chapa, 2 Cats Talking
6.2.13. 2024년[편집]
1월: 2.2 Lobotomy, Dancing Toothless, boykisser[97], Verbalase, Dancing Wolf Meme, Bling-Bang-Bang-Born, Looksmaxx, virtual sin forgiveness, Chipi Chipi Chapa Chapa
2월: Bling-Bang-Bang-Born, Verbalase, Raiden In Fortnite, Don't leave so soon, beat da koto nai, gegagedigedagedago, Bye-Bye(mewing), Ha Cha Cha, Girlfriend, Human... I remember you're, Life or bath for dry cat, Oiia Oiia Cat
3월: Lovely Loco, Pokémon Dance, Arona Dance, Sad Hamster, Cotton Eye Joe, Left or Right, You are my sunshine, how ~ works, tengetenge, Pure Pure, SLAY[98], Piltos be Like, Tenge Tenge, Pedro Pedro Pedro Racoon, Beautiful Things, Masha Ultrafunk
4월: Pedro Pedro Pedro Racoon, I Believe In Joe Hendry, You are my sunshine, More Herta Please!, Smoking Duck, Glazing, Duck Song, A song about bread, gwimbly, America Ya, Showers are too sensitive, Daddy's home, oi oi oi, Around the World, Big shoe lmfao, If we being real, silly billy, I Go Meow
5월: I told you Dave, I never lose., Around the World, LeBron James, Scream if you love, english isn't my mother language, BBL Drizzy, Miku Miku Beam, Ei ei ei I'm on vacation, fun 2 rhyme, Open the Door, jodellavitanonhocapitouncazzo, Miaw miaw miaw, Let's go Gambling!, Tyla Dance
6월 : Shikanoko Dance, Dithching School, English or Spanish, August 12 2036, the heat death of the universe!, E.T is An Alien, FUn 2 rhyME, She’s my Alibi, Not My Problem
7월 : Not My Problem,Superman Starman, August 12 2036, the heat death of the universe!, Freaky bob, Fight… Fight Trump, cute girl simply[99], Little John[100][101], Go-Getters, Wizard gnomes, aura, Daten Route, Attempted assassination of Donald Trump, Army Dreamers, Hai Yorokonde, You think you just fell out of a coconut tree?, Inside Out 2, Ignore all previous instructions, Republicans are weird, Shadow Wizard Money Gang, ei ei ei, now a cow pretending to be a man, Pikki Pikki Dance, Big Dawgs
8월 : Pikki Pikki Dance, Pepe Dance, Paris Olympics Opening, Olympic Shooter Anime, Dog Fish, Crucified Minion, Bye Bye Bye, Have Some Chocolate Milk, Honey Pie, Brazilian Miku, Maitake Dance, Duolingo Meme, I JUST WANNA BE YOUR SYMPHONY, Emergency Dance, Conan Dance, Turkish Shooter, Why so serious? ai
9월 : If you Dance I'll dance, Brazilian Miku, You think you just fell out of a coconut tree, Concord, Astro Bot, I am Steve, The Debate Was Cooked Again: debate trump vs harris, Nikocado Avocado Two steps ahead, It's time to play the game, Symphony, Deep Thoughts With The Deep, mio mao lalalala, Bouncing Yaris, MrBeast Rizz Dance, Wolverine, Mentality, doodle dance
10월 : I am Steve, Deep Thoughts With The Deep, I Just Lost My Dawg, mio mao lalalala, Joker: Folie à Deux, Never KILL Yourself, Maitake Dance, Ombrinho, Diddy, Baby Oil, She Know, Thick of it, doodle dance, APT., Hawk Tuah, Burning Desires, One step at a Time
11월 : APT., Blender Cloth Simulation VS Faces, Hawk Tuah, Burning Desires, Battle forte, Election, That feeling when knee surgery is tomorrow, everyone looks awful from underneath, Low taper fade, Queen never cry, The Ki Sisters, wait you don't love me like i love you
12월 : Apple dog, Queen never cry, The Ki Sisters, Chill guy, Chinese Rapping dog, Korean Martial Law, Low taper fade, Luigi Mangione[102], Hai Phút Hơn[103], Sigma Boy, 5x30, king von, Locked In Alien: What Is the Next Step of the Operation?, Squid Game 2
6.2.14. 2025년[편집]
1월: 5x30 5min/day, Sigma Boy, Squid Game 2[104], Chopped Chin, I bought a property in Egypt , Sonic ai, Chill guy
2월:
3월:
4월:
5월:
6월:
7월:
8월:
9월:
10월:
11월:
12월:
7. 기타 목록[편집]
상세 내용 아이콘  자세한 내용은 밈(인터넷 용어)/기타 문서를 참고하십시오.
8. 여담[편집]
레딧에서 밈이 많이 유통되는 편이며, 이를 정리해서 유튜브에 배급하는 Memenade나 Cowbelly Studios, BENBROS, Lessons in Meme Culture[105], Meme Zee[106], Dake 등의 채널들이 있다. 또 대형 채널로 Kracc bacc이 있다.
'How to Pronounce Meme'이라는 이름의 유튜브 동영상을 보면, 해당 발음은 /miːm/(/밈/)으로 발음되고 있다. 하지만 댓글란에서 일어나는 병림픽이 문제. '미미', '멤메이', '메메', '메미', '메이메이' 등 천차만별의 의견이 있는데, 해당 단어는 '밈'으로 발음하는 게 옳다. 자세한 내용은 밈 문서의 발음 문단 참고.
영어판 위키백과에는 밈을 모아둔 분류도 있는데, 2013년까지 무려 550개가 넘어가는 목록을 보유하고 있다. 또한 밈 백과사전이라고 할 수 있는 사이트 'Know Your Meme'도 있다.
The GAG Quartet라는 밴드가 그 동안에 유행하던 밈들을 전부 모아 리믹스한 적이 있으며, 여러 밈들을 매시업한 영상 역시 쉽게 찾아볼 수 있다.#1 #2 위저의 〈Pork And Beans〉 M/V는 아예 당대 인터넷 밈을 주제로 삼아 유튜버들이 직접 참여했다. 앨런 베커의 한 작품에서도 밈을 비롯한 여러 영상이 잠깐 나온 적이 있다.
나무위키에서는 보통 '누군가에게 어떤 인식을 주입시키거나 용례를 파급시키기 위하는 밀어주기'라는 뜻으로 통용된다. 한국의 대표적인 예시로는 '못 간다고 전해라', '조세호의 결혼식 불참 드립', '원더걸스의 Tell me 패러디' 등이 있다. 이러한 밈이 남들에게 "아 노잼인데 저거 자꾸 왜 해?"[107]와 같은 반정서를 일으키는 경우에 억지 밈이 된다.
유럽연합이 자체 저작권법에 인터넷에서 꾸준히 논란이 많았던 제13조(속칭 'Meme Ban')를 추가하는 수정안이 의원 투표를 통과하면서 EU 안에서 밈이 불법이 되었다. 원작의 저작권이 침해된다는 게 이유이다. 하지만 오히려 이런 유럽을 비꼬는 밈이 더 만들어진다. 레딧에서는 EU를 밈으로 만들어 EU를 금지시켰다. 그리고 제13조가 통과되면서 매우 기뻐한 Axel Voss도 밈으로 만들어서 금지시켰다.
최초의 밈이 1921년에 나왔다는 유머도 있다. 내용은 플래시라이트를 키면 잘생겨질 줄 알았지만 현실은 시궁창이라는 내용.
2018년 11월 11일 방송된 KBS 도전 골든벨 평택여자고등학교 편 50번 마지막 문제의 정답으로 나왔으며 여기서 골든벨 우승자가 탄생하였다.
밈에도 저작권이 있다. 밈을 NFT화하거나 상업적/비상업적 용도로 사용하려면 저작권자의 허락을 받아야 한다.#
한국에도 밈을 전문적으로 소개하는 팟캐스트 방송이 있다. 밈스터치가 대표적이다.
나무위키에서 인터넷 밈 패러디 영상의 자체 등재기준은 없지만 토론을 통해서 종류에 따라 다르지만[108] 대부분 조회수 10만회 이상이 되어야 등재가 가능하도록 한다. 유튜버 문서의 등재기준이 구독자 3만 명 이상인 것처럼 조건 미달의 영상을 통한 채널홍보로 인해 문서가 폭주하는 것을 막기 위한 일종의 불문율이다.
9. 관련 문서[편집]
4chan
9GAG
Fail Blog
Memenade
MLG(밈)
SiIvaGunner
SMG4
YouTube Poop
YTMND
게리 모드
뇌절
레딧
리처드 도킨스
밈
소스 필름메이커
오퍼레이션 블랙 레이지
인터넷 용어, 인터넷 은어, 인터넷 유행어: 2020년부터는 이와 관련된 용어들은 거의 밈으로 대체되었다.
캐릭터 밈
억지 밈(한국)
인간 관악기(한국)
트위치/대한민국/밈(한국)
필수요소(한국)
네타(일본)

[1] 물론 인터넷의 특성상 현재 유행하는 밈의 상당수는 짤방이라고 부를 수 있긴 하지만.
[2] 2020년 3월경 규정 개정으로 금지되어 옛말이 되었다.
[3] 대부분의 유행어도 이 짤방들의 자막에서 유래했다.
[4] 유명인에게서 한정되는 것도 아니고 어느 날 우연히 일반인이 올려놓은 게시물이나 영상같은 것이 뜬금없이 유튜브 알고리즘 같은 원인으로 뜨면서, 이게 밈으로 재조명될 수도 있다. 즉, 이 문서를 보는 당신도 무언가 올려놓은 게 있다면, 그것이 어느 날 갑자기 밈으로 쓰여도 이상하지 않다는 것.
[5] 사실 정확하게는, Nyan Cat에 삽입된 노래는 하츠네 미쿠가 부른 게 아니라 UTAU인 모모네 모모가 부른 버전이다.
[6] 해외에서는 meme review라고 부르기도 한다.
[7] 빅맥송은 그냥 사진만 적당히 붙혀넣기한 다음 슬라이드쇼하면 그만이지만 제로투 댄스는 원본이 애니메이션이다보니 그 춤을 일일이 한장한장 그리거나 리깅을 해서 애니메이션으로 만들어야 한다. 그때문에 웬만한 편집 프로그램으로는 불가능하고 클립스튜디오나 어도비 애프터 이펙트 정도는 동원해야 한다. 그냥 본인이 직접 춤추는 영상을 올린다는 간편한 방법이 존재하지만 본인의 외모가 받쳐주지 않는다면 쉽게 묻힌다.
[8] 이쪽은 아예 런닝맨에서 소개되는 밈은 "사망 선고" 라며 자학 개그까지 선보였다.
[9] 정확히는 구미권의 밈과 다르게 동아시아권의 밈은 영상 내용을 개조한 물건이나 음MAD 등의 꽤 정교한 2차 창작을 위주로 발전하다보니 이에 밈 관련 인물에 대한 캐릭터성이 형성, 이를 토대로 다시 2차 창작이 제작되면서 밈 자체에 대한 팬덤이 형성되어 같은 밈이 10년 이상 소비되는 경우도 드물지 않다. 다만 2020년대 이후로는 동아시아권도 서양 밈 자체가 유통되거나 토착 밈이여도 편집 효과 활용 내지는 짧은 패러디 등이 주가 되면서 서구화되고 있으며, 이 시점 이후로 발생하여 나름 장기간 회자되는 밈은 일본의 와카 정도에 국한된다.
[10] 실제로 페페 더 프로그는 대안 우파의 상징이 되기도 했다. 이 비화는 '인싸를 죽여라'라는 책으로 출판되기도 했으며, 당연히 이 책의 시각을 옹호하며 밈을 비판하는 칼럼 역시 기재되었다.
[11] 논리적으로 사고하지 않고 밈의 흐름대로 사고한다는 뜻이다. MBC 2020 도쿄 올림픽 개막식 중계방송 사진 및 문구 논란이 일자 위근우가 이 논리를 들어 다시 한 번 나무위키를 비판하기도 했다. 아이러니하게도 이런 밈같은 용어를 애용하는 이들이 이슈를 대하는 방식이나 이러한 사회과학 전문 용어를 습득하고 사용하는 방식 역시 (인터넷 밈이 아닌)광의의 밈과 비슷한 양상을 보인다. 심지어 '기립하시오 당신도' 같은 그들만의 밈을 애용하면서 밈을 비난하는 이중잣대를 보이는 경우까지 있다.
[12] 참고로 이 베이비 차차는 오리온에서 출시한 과자 베베의 사이버 아기 '짜루'로도 패러디되었다.#
[13] 얘는 해외 파생밈도 아니고 국산 밈이다.
[14] 특히 Bowsette
[15] 물론 커비는 자체적으로 순화시킨다.
[16] 다스 시디어스를 연기한 이언 맥디어미드의 찰진 발음으로 Dewitt!하는 게 밈이다.
[17] 라더스의 몬 칼라마리족 부관으로 이름은 Shollan이다.
[18] 주로 옆으로 미끄러지는 영상에서 데자부! 하는 타이밍에 맞춰 편집하는데, 드리프트가 문제가 아니라 각종 그냥 미끄러지기만 하는 영상에 때려박기도 한다.
[19] 간혹 가다가 폭주하는 자동차 영상에 유로비트를 넣기도 한다. PUBG나 배틀필드에서 자동차를 운전할 때 일부러 이 노래를 트는 사람이 있기도 하다.
[20] 러시아의 특수부대
[21] 접착식 집속탄. 간단하게 설명하자면 설치된 표면 뒤로 수류탄 5발을 쏜다.
[22] 세계 무역센터 테러사건이 화재가 되었던 당시 전국 초중생 사이에서 유행하던 오사마 빈 라덴 찬양송이다. '내가 제일 존경하는 오사마 빈 라덴~ 나도 커서 이 다음에 테러범이 되고 말거야~'로 시작한다.
[23] ㅋㅋ 같은 초성어는 이전에도 존재했으나, 크레이지 아케이드의 인기와 더불어 수많은 초성어가 탄생되기 시작한다.
[24] 이 때부터 엽사를 필두로 한 '글자 줄여 말하기'가 본격적으로 시작되었다.
[25] 애니콜 핸드폰 CF에서 나온 멘트이다.
[26] 후에 등장하는 뚝배기 유행어의 고착화에도 도움을 주었다.
[27] 넌 나에게 ~~를 줬어 등의 바리에이션으로 구성된다.
[28] 일명 경인대첩이라 불리는, 사상 최대규모의 한일 사이버 유저 대첩이다.
[29] 사실상 고자라니 다음으로 오래 살아남은 밈이다.
[30] 영화 아저씨의 명대사. 자매품으로 '이거 방탄유리야 이 개새끼야'가 있다.
[31] 코로나 19로 인해 자택근무가 일상이 되었다 보니 다른 시기에 비해 많은 인터넷 밈들이 탄생하였다.
[32] 다만 같은 해의 10월 28일에 행정안전부에 의한 차단 조치를 당하면서 사실상 데이터 말소에 가까운 수준으로 사라졌다.
[33] 사회 문제 해결엔 별 도움이 안되고 그저 황당함에 웃음만 나온다는 점에서 밈이 되었다.
[34] 단계적 일상회복이 진행되면서 유행하는 인터넷 밈의 빈도가 작년에 비하면 많이 줄어들었지만 틱톡과 유튜브 쇼츠의 영향으로 다양한 밈들이 쏟아져 나오고 있다.
[35] 처음 등장한 것은 10월이지만 본격적인 유행은 11월부터였으며, 12월 카타르 월드컵을 계기로 전국민적으로 확산되어 사실상 22년의 마지막 순간까지를 장식한 밈이다.
[36] 릴스를 비롯한 숏폼 플랫폼의 영향으로 해외발 밈도 많이 수입되었다.
[37] 영화 내부자들에서 이경영의 배역 장필우의 일명 꼬탄주 대사에서 나온 밈이다. 하지만, 실제로는 이 대사가 전혀 나오지 않지만 일부러 대사가 있는 것처럼 연기하는 것이다.
[38] 하이디라오 춤이라고도 불린다.
[39] "실패하면 XX, 성공하면 XX 아입니까!", "그 이왕이면 XX(이)란 멋진 단어를 쓰십시오!"
[40] 1월 달 유행했던 밈의 확장판으로, 멋진 외제차가 드리프트를 하거나 호날두가 슈팅연습을 하는 등 괴이한 바리에이션이 많이 생겼다. 수많은 연예인들이 이 챌린지에 참여하면서 유행하게 되었다. 인스타그램에 원본이 있으며 뉴스에도 나왔다.
[41] 이 음원에 곰돌이 푸가 강남스타일을 추는 영상이 유행중이다.
[42] ~~유전자 섞을래?, 우리 집 코끼리 보러갈래? 귀엽네? 스껄할래? 작업실 놀러올래?
[43] 주요 이메일 체인들을 통해 퍼젔다고 한다
[44] Blend S 인트로 영상을 바탕으로 했다.
[45] Moth는 Møth와 Möth, Lamp는 Lämp와 Læmp로 자주 쓰인다.
[46] No Nut November = 11월 금딸 챌린지. 바로 뒤 12월은 Destroy Dick December라고 해서 11월에 하지 못한 것을 마음껏 푸는 달이다.
[47] 이름은 Duo.
[48] 벨 델핀이 목욕물을 온라인상으로 판 것이 밈화되었다.
[49] 주인을 베개로 깔아뭉개는 강아지
[50] 가장 오래된 영상으로 추정되는 Yandere Dev 버전.
[51] 용과 같이 시리즈의 ばかみたい를 부른 영상을 사용해 딥페이크로 다양한 사람들이 부르게 하는 것이다.
[52] 우주비행사 2명이 서있는데 지구가 전부 미국 오하이오 지역으로만 가득 찬 걸 보면서 미국 국기를 붙인 비행사가 "잠깐, 전부 오하이오였어?"라며 매우 황당해하는데, 뒤의 오하이오 주기를 붙인 비행사가 그의 뒤통수에 권총을 겨누고 "항상 그래왔다"라며 쏴 죽이려는 장면이다. 사실 처음에는 링크처럼 실사 그래픽 우주비행사가 아니고 익명의 4chan 유저가 어설프게 그린 그림으로, 비교대상도 미국이었다. 뜬금없이 오하이오 지역이 된 이유는 2016년~2017년 중반에는 오하이오 지역 자체가 밈이었기 때문이다. 오하이오를 중심으로 미지의 위험이 일어나거나 오하이오가 전 세계를 침략할 거라며 그로 인해 오하이오를 폭파시키거나 말살하겠다는 등의 밈이 존재했었다. 어찌 보면 해당 드립의 연장선인셈. 단 해당 영상 링크에서 보듯이 변형 답안들도 상당히 많이 존재한다. 대표적인 변형 밈 영상으로는 이것으로 우주비행사가 전부 지구였냐며 황당해하고 아니 12년동안 훈련 쳐 받아놓고 우주비행사가 어떻게 그걸 모르냐, 개리 이 개병ㅅ 이라고 화내는 다른 우주비행사가 압권 또한 헌터×헌터 애니의 음악 중 하나인 Kingdom of Predators가 주력 배경음으로 쓰이는 탓에 졸지에 이 음악도 조명받고 있다.
[53] 2020년 하반기 어몽어스란 게임이 큰 인기를 끌게 되면서 자연스레 밈이 많아지기 시작했다
[54] 인도네시아의 인터넷 회사의 요금제 광고이다. 12살 어린이가 만든 파워포인트처럼 난잡하게 돌아다니는 텍스트와 어색하게 서 있는 남자, 무엇보다도 신나고 중독성 있는 음악이 특징. Stickbug 밈과 비슷한 Rick Roll류의 밈이다.
[55] 발매 후 각종 버그와 악평으로 밈이 재유행했다.
[56] 배달의민족이나 요기요와 같은 음식 주문 플랫폼인데, 원본 광고 영상을 보면 싫어요가 훨씬 많다.
[57] 단순한 밈을 넘어 뉴스에도 보도가 되는 등 사회 현상이 되었다.
[58] 신나는 동물농장
[59] 아동용 컨텐츠가 불쾌한 골짜기로 이루어진 저퀄 애니메이션에 각종 마약, 성적 컨텐츠를 접할 수 있는 것에 대한 내용으로 이루어져 있다.
[60] 태어나고 싶은, 또는 태어나고 싶지 않은 나라가 있어 아기가 신에게 간절히 기도를 하는데 결국 웃긴 방향으로 실현된다는 내용이다. 예를들면 일본에서 태어나고 싶어 하는데 하필 태어난 날이 1945년 8월 원폭투하날 일본이라는 둥... 후에는 '둘 중에 한 곳에는 태어나지 않게 해주세요' 하는데 두 특징을 모두 가진 곳에 태어난다던가 하는 걸 봐선, 그냥 생까는듯.
[61] 동숲에 클레오라는 주민이 Hentai에 나온게 인기가 많아 밈이 되어버린것이다. 물론 원본은 19금이다.
[62] LowTierGod이 온갖 욕설을 하면서 '넌 그냥 자살해야 돼, 지금 당장!' 정도의 말을 뱉은 것인데... 참 주옥같은 명대사처럼 자리잡아버렸다. 주로 번개와 합성한 걸 자주 쓰는 편. 지금도 자주 보이지만, 유행 당시에는'you should' 으로 시작하는 문장에 번개 이모지를 치는 식으로도 쓰일 정도로 인기가 많았다.
[63] 크리스마스가 1주밖에 안 남았어요! 하는 여자의 영상이긴 한데... 말투가 정말로 영혼이 없고, 크리스마스가 1주밖에 안 남았다는 말만 계속 반복한다. 'I am so happy about this information' 이라는, 한국말로 치면, '이 일에 대해 정말 기쁘네요' 라는 이상하게 격식 차린 말은 덤. 쓰임은 대부분, christmas 부분을 다른 사진을 넣고, 크리스마스라고 말할 때마다 그 사진에 맞는 소리를 반복해서 보여주는 것. 주로 밈 효과음이 사용돼지만, 섬광탄이나 나중에 유행하는 demon core같은 재밌는 편집도 있었다.
[64] 사울 굿맨의 3D 모델 영상은 아니고 이미지 가지고 3D로 만든듯한 느낌이다. 10월 영상이나 유행은 이때쯤 한 것으로 보인다. 베터 콜 사울 오프닝에 변치 않고 응시하는 표정이 압권. 사실상 브레이킹 배드 시리즈가 2022년에 밈으로서 유행하게 된 시초라고 볼 수 있겠다.
[65] 낚시 영상에 편집으로 난입하는 밈이 가장 자주 보인다. 다만 이런 편집 밈은 원래 유서가 깊은데, 여기에 보여도 너무 자주 보이는 메탈 기어 라이징 관련 밈들이 합쳐지다 보니 메탈 기어 라이징 관련 밈 전부를 보는 시각이 안 좋아지기도 했다. 꽤나 오래 유행하기도 했고.
[66] 이와 관련해서는 stapsi ah (DEAF KEV - Invincible), Hi hi hi (JPB - High)가 있다.
[67] 파란색 랍스터 사진에 토카타와 푸가를 틀어놓은 밈. 별것도 아닌 랍스터 사진이 뭔가 심각해 보이는 게 포인트. 그 외에도 랍스터의 패션을 바꿔 놓고 그에 어울리는 토카타와 푸가 리믹스를 넣는 등 바리에이션이 있었으나, 대부분 원본 또는 귀갱버전 또는 4K 버전이 디스코드 등지에서 맥락없이 보내지면서 쓰였다.
[68] 원본은 언더테일의 Fallen down을 Moonbase alpha에서 부른 버젼을 마치 자신이 부르는 것처럼 찍은, 조금 괴기한 필터를 낀 영상인데, 여기에 캡션으로 My reaction to that information, 즉 '그에 대한 내 반응' 이란 말을 넣은 게 유행을 탔다. 이후에는 그 캡션이 밈으로 유행해서 다른 사진에 끼워넣은 것들이 온라인에서 대화할 때 사용되는 중.
[69] 작년의 제로투와 유사한 Cringe한 밈.
[70] 캐릭터들 능력치를 비교하고 누가 더 위인지 비교하는 밈. 한국에서는 주로 외모지상주의, 김부장, 더 복서 등 웹툰이 주 희생양이다.
[71] 보물섬(1988년 애니메이션) 항목 참조.
[72] 버질이 영상 도중 특정 순간에 화면을 깨부시는 내용이다. bloodshed 관련 밈과 유사.
[73] 이것도 Vergil Status와 유사한데, 초반에 여자 영상으로 낚시를 한 뒤 하드스타일이 흘러나오며 시간낭비 그만하고 자기 발전에 힘쓰라는 영어 자막이 나온다.
[74] 흰 수염이 One Piece Is Real이라 외치고 칸예 웨스트의 Dark Fantasy가 흘러나오는 괴상한 밈. 원본은 한 트위터 유저가 흰수염에 사진에 거대한 남근을 단 합성사진으로부터 시작됐으며 검열문제로 원래 사진 모습대로 이어지다가 갈수록 여러 바리에이션이 늘어났다. 베터 콜 사울에서 하워드 햄린을 연기한 패트릭 파비앙이 이 밈을 따라하면서 유명세가 커지게 되었다.
[75] Bad Bunny의 neverita를 bgm으로한 짧은 요리 동영상 중간에 게이 키스, 동물들이 입을 맞추는 짧은 영상을 끼워넣는다. 가끔 기출변형으로 순서가 반대가 된다던가 그냥 멀쩡한 요리 영상만 나온다던가 위험한 영상이긴 한데 그 방향이 다르다던가 같은 식의 영상도 나온다.
[76] 영상이 기므로 대략 줄거리를 요약하자면, 초반 악역이던 F가 다른 알파벳들을 납치하고 N을 포함한 몇몇 알파벳들이 이들을 구하러 간다. 그러나 F가 악역이 된 이유는 사실 F가 먼저 다른 알파벳들에게 왕따를 당하여 흑화한 것이었는데...
[77] 덩달아 20년도의 그 게임이 또 떠올랐다.
[78] 11월 중반부터는 파생 밈으로 'can't even play minecraft in Ohio 💀' 라는 밈이 유행중. 참고로 해골 이모지는 상황이 막장이거나 충격적일 때 주로 사용된다. 마인크래프트의 온갖 논리와 원리를 비튼, 이상한 게임플레이를 보여준다. 오하이오는 마인크래프트도 일반적이지 않다는 게 웃음 포인트. 메이드 인 헤븐을 변형한 'Made in ohio' 도 있다.
[79] 위에 캡션에 어떤 상황인지 적어두고, 총 등의 작동원리 영상을 밑에 놓은 밈. 배경음악이 공통적으로 Mississipi Queen이라서 이름이 이렇다. 위의 상황은 보통 누군가 짜증나게 구는 상황. 즉 그냥 쏴버리는 것을 표현한 것. 다양한 캡션이 있었는데, 공통적으로 쏘는 건 극단적인 방법이면서 범죄행위지만, 짜증나거나 이익이 되는 상황(100% 할인 쿠폰이라거나.)이다. 물론 가끔 가다 완전히 다른 상황도 보이는데, 유명한 것으로 하프라이프에서 타우 캐논을 과충전하는 장면을 SFM 유튜버가 만든 것. 짜증난다고 단순무식하게 쏴버리는 행위와 천천히 총 작동원리를 보여주는 것이 웃겼던 것인지 꽤 유행했다.
[80] 얼굴에 이상한 필터를 낀 남성이 좋은 걸 알려준다고 유혹하려다 갑자기 분위기가 어두워지며 남성이 기괴하게 웃으며 춤추는 낚시용 밈. 원본은 공주와 개구리. 사실 전에도 이 음악을 이용한 밈들이 조금 있었으나 이걸로는 상당히 유행했다. 위의 My reaction과 같은 사람이다.
[81] 그냥 1시간 동안 감자칩이 제자리에서 회전하는 것을 보여주고 배경음으로 조잡한 음질의 Funkytown이 재생되는, 별 것 없는 영상이지만 이 영상의 댓글창은 어째서인지 이 영상을 예술적인 측면에서 극찬하는 댓글들이 주를 이루고 있는 것이 포인트. 자매품으로 치킨 너겟도 있다.
[82] 원곡은 불가리아 노래인 FIKI - Chupki V Krusta. 춤추는 사람은 야신 센기즈(Yasin Cengiz)라는 터키인이다. 별의 별 리믹스가 다 있다.
[83] 댓글을 보면 죄다 "The waffle house has found its new host" 로 도배되어 있는데, 이것 역시 밈이다.
[84] 웬즈데이 아담스가 춤추는 장면과 레이디 가가의 bloody mary가 재생되는 영상이 쇼츠나 틱톡으로 퍼졌다. 옆의 Brr skibidi와 합친 버전.
[85] 과거에 게리모드에 누군가 저폴리곤 고양이 모델을 올렸는데, 점점 인기를 끌다가 이번에는 제대로 히트를 쳤다.
실제 이름은 Jess라고 하며, 맨 처음 나왔던 게리모드 애드온에선 실제로 있는 Dingus라는 비슷한 고양이와 헷갈린건지 그 이름을 사용해서 그렇게 부르는 사람도 있으나, 이 모델을 사용한 '들고 다닐 수 있는 고양이 맥스웰' 애드온이 훨씬 더 크게 유행하면서 맥스웰이라는 이름이 붙었다. 일본의 '우니' 라는 고양이와도 비슷하게 생겨서 헷갈리기도 하는데, 밈으로 쓰일 때 이름은 'Big boobs', 즉 큰 찌찌.(여담으로, 이는 또다른 밈인데, 겉보기엔 위험해 보이진 않지만 검색하면 야짤이 나온다던가 눈갱짤이 나온다던가 등 조금 위험한 검색어를 '이 고양이 이름은 ~~입니다, 검색해보세요' 식으로 위장한 밈이 조금 유행한 적이 있었다. 그러나 한두 번쯤 당해보면 너무 뻔히 보이는 함정인지라, 이를 비꼬아서 '얘 이름은 게이야동임 구글에 쳐보셈' 같은 밈이 나오게 됐고, 우니 사진의 경우는 '얘 이름은 Big boobs임' 이라는 밈이 나오게 됐다. 그런데 이게 상당히 유행해서, 그냥 이름이 되어 버렸다.)워낙 똑같이 생겼다 보니 그냥 둘 다 서로의 이름으로도 부르는 경우가 많다.
[86] 배경음악의 가사를 쓴 것으로 보인다.
[87] 이전에도 MauzyMice의 애니메이션이 다른 밈으로 유명했으나 이때부터 Boykisser란 명칭이 처음 쓰이면서 사실상 스테디셀러로 자리잡았다.
[88] 원곡
[89] 게임 OST나 앨범, 특정 그룹의 곡들을 틀어놓고 그 경험을 다양한 밈 영상이나, 기분을 느낄 수 있는 영상으로 설명하는 밈. 애초에 같은 음악을 애청하는 사람들이 주로 보는 영상이기에, 공감된다는 느낌으로 상당한 인기를 끌었다. 데프톤즈가 주로 보이며, 이것이 밈의 발원지인 것으로 보인다. 게임 중에는 언더테일이나 델타룬이 자주 보인다.
[90] Mustard의 Ballin이 BGM으로 깔려서 뮤직비디오 댓글엔 죄다 밈 얘기 밖에 없다. #
[91] AI 목소리로 역대 미국 대통령인 조 바이든과 도널드 트럼프가 게임하면서 음성 채팅하는 역할 놀이를 하며 주로 바이든은 츳코미 트럼프는 보케 (여기서 트럼프는 바이든을 '슬리피 조'로 부른다.) 하고 있다. 또한 바이든과 트럼프외에도 버락 오바마, 조지 부시랑 같이 놀고 있다. 게임 말고도 어떤 애니나 영화, 드라마가 최고인지 토론을 하는 여러 바리에이션이 있다.
[92] 아이젠 에딧 버전
[93] 릴스 등에서 이 밈이 유행한 2023년의 4년 뒤인 2027년에 유행할 밈이 될 것이라는 밈도 있다. 또 사막이나 정글 등 서식지도 다양하다.
[94] ㄱGyatt나 Rizz나 Fanum tax 같은 단어는 미국의 유튜버 및 스트리머 Kai Cenat 관련 밈에서 유래한 단어다. 이외에도 원래 밈에 나오는 skibidi는 Skibidi Toilet에서, sigma는 Sigma 밈에서 따온 것이다.#
[95] 레알 마드리드에서 골을 폭격하고 있는 주드 벨링엄의 세리머니를 밈으로 만든 것이다. 벨링엄처럼 팔을 넓게 벌리는 작은개미핥기 영상과 해설을 같이 삽입하여 올린다.
[96] 영화가 개봉하면서 재유행하였다.
[97] Chipi Chipi Chapa Chapa로 인한 재유행이지만 사실 여전히 충분한 인기를 누리고 있긴 했다.
[98] 음악에 맞춰 펀치 갈기는 밈
[99] KFC도 올렸다
[100] 주로 집 확장 건설에 관련된 숏폼이다.
[101] 해당 숏폼에서 자주 사용하는 Galvanized square steel, Eco-friendly wood등도 밈화되었다.
[102] 유나이티드헬스케어 CEO 총격 피살 사건과 연관이 있다.
[103] 제로투 댄스라는 언급이 없을 뿐이지 제로투 댄스가 맞으며 몇번째로 또 재유행했다.
[104] 주로 성기훈이 "난 이 게임을 해봤어요!"라고 소리치는 장면이 사용된다. 물론 영어 더빙버전으로
[105] 전자는 실제 밈을 만드는데, 21st century humor마냥 정신없는 편집과 효과음이 특징. 인기가 꽤 있는데다 애니메이션 스타일도 PNG 사진에 관절 넣어서 대충 움직이게 하는 방식이라 호불호가 갈릴 수 있다. 후자는 실제로 가이드처럼 현재 유행하는 밈에 대해 알려준다. 정보도 빠릿빠릿하고 꽤나 논리적인 유행 원인 분석이나 설명, 유래 등도 있어서 볼만하다. 가끔 크게 유행하지 않은 밈을 알려줄 때도 있다.
[106] Benbros와 유사.
[107] 보통 이외에도 '할 거면 니들만 해.', '만든 사람은 자기가 획기적이라 생각하겠지.' 등의 반응이 지배적이다. 일본어의 고리오시의 결과에서도 비슷한 반응을 낳기도 한다.
[108] 관짝춤처럼 전 세계발로 흥행한 경우에는 50만회까지 등재기준이 올라간다.
CC-white 이 문서의 내용 중 전체 또는 일부는 다른 문서에서 가져왔습니다.

밈(인터넷 용어)
밈(인터넷 용어)
밈(인터넷 용어)
삼청동맛집 한우전문 설렁탕
삼청동맛집 한우전문 설렁탕
map.naver.com/v5/entry/place/1788911444
아직도모르세요?일본잡지 책에도 소개 장차관님 밥집 직접 담근깍두기 설렁탕 환상궁합
잠실 평양냉면 전문점
잠실 평양냉면 전문점
m.store.naver.com/places/detail?id=1973334226
한우 육수 평양냉면과 다양한 비빔냉면에 전립투 샤브까지 드실 수 있는 냉면 전문점
한국다이아몬드센터 서울금매입
한국다이아몬드센터 서울금매입
www.koreadiamonds.com
카톡상담센터예약출장예약고객스토리
서울금매입 (사)한국다이아몬드협회 인증거래소, 현장감정 및 당일시세 매입
크리에이티브 커먼즈 라이선스
이 저작물은 CC BY-NC-SA 2.0 KR에 따라 이용할 수 있습니다. (단, 라이선스가 명시된 일부 문서 및 삽화 제외)
기여하신 문서의 저작권은 각 기여자에게 있으며, 각 기여자는 기여하신 부분의 저작권을 갖습니다.

나무위키는 백과사전이 아니며 검증되지 않았거나, 편향적이거나, 잘못된 서술이 있을 수 있습니다.
나무위키는 위키위키입니다. 여러분이 직접 문서를 고칠 수 있으며, 다른 사람의 의견을 원할 경우 직접 토론을 발제할 수 있습니다.

s25
검은 수녀들
김문수
나재견
전한길
Lck
치매
윤석열
이세돌
손흥민
s25
하나은행 K리그1 2025/겨울이적시장
43초 전
단간론파 시리즈
43초 전
일본 130번 국도
45초 전
Warp Corporation
49초 전
카미시로 츠루기
54초 전
보너스 배틀
55초 전
비몽(인터넷 방송인)
1분 전
하마 학살
1분 전
SPRUNKI/모드
1분 전
요정왕 쿠키
1분 전

234명 성착취 '자경단' 총책 검찰 넘겨져…질문에 묵묵부답
갤 S25 사전판매…단통법 아래 마지막 공시지원금 5만~24만원대
"서부지법 습격자 끝까지 찾는다"…경찰, 연휴에도 수사
"타 여성과 관계" 무릎 꿇은 남편, 이혼 조정 중 책상 '쾅'…서장훈 "여봐요" (이혼숙려캠프)[종합]
뉴진스 아닌 뉴진스?…"새 활동명 공모"vs"계약 위반" 팽팽 [종합]
더 보기



namu.wiki
Contáctenos
Términos de uso
Operado por umanle S.R.L.
Hecho con ❤️ en Asunción, República del Paraguay
Su zona horaria es Asia/Seoul
Impulsado por the seed engine
This site is protected by reCAPTCHA and the Google Privacy Policy and Terms of Service apply. This site is protected by hCaptcha and its Privacy Policy and Terms of Service apply.
"""


  
        
    if text:        
        paragraphs = text.split("\n\n")[:-1] if "\n\n" in text else text.split("\n")
    else:
        paragraphs = []    
        
    # FAISS 벡터 스토어 생성
    with st.spinner("벡터 스토어를 생성하는 중..."): 
        # convert to Document object (required for LangChain)
        documents = [Document(page_content=doc, metadata={"source": f"doc{idx+1}"}) for idx, doc in enumerate(paragraphs)]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)    
        
        vectorstore = FAISS.from_documents(documents=splits, embedding=st.session_state.embedding)
        
    return vectorstore

# RAG using prompt
def rag_chatbot(question):
    context_docs = st.session_state.vectorstore.similarity_search(question, k=2)
    # for i, doc in enumerate(context_docs):
    #     st.write(f"{i+1}번째 문서: {doc.page_content}")
        
    context_docs = "\n\n".join([f"{i+1}번째 문서:\n{doc.page_content}" for i, doc in enumerate(context_docs)])

    # prompt = f"Context: {context_docs}\nQuestion: {question}\nAnswer in a complete sentence:"
    prompt = f"문맥: {context_docs}\n질문: {question}\n답변:" #context_docs는 핵심이다.
    # response = gemini_model(prompt)
    
    response = st.session_state.model.generate_content(prompt)
    answer = response.candidates[0].content.parts[0].text

    print("출처 문서:", context_docs)
    return answer, context_docs


# Streamlit 세션에서 모델을 한 번만 로드하도록 설정
# 1. gemini model 
if "model" not in st.session_state:
    st.session_state.model = load_model()

# 2. embedding model
if "embedding" not in st.session_state:
    st.session_state.embedding = load_embedding()

# 세션의 대화 히스토리 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "topic" not in st.session_state:
    st.session_state.topic = ""


# 1. 이 주제로 Vectorstore 만들 문서 가져오기
topic = st.text_input('찾을 문서의 주제를 입력하세요. 예시) 흑백요리사: 요리 계급 전쟁(시즌 1)')

if st.button('문서 가져오기'):
    if topic:
        vectorstore = create_vectorstore(topic)
        st.session_state.vectorstore = vectorstore    
        st.session_state.topic = topic
    else:
        st.warning('주제를 입력해라', icon="⚠️")
    
if st.session_state.topic and st.session_state.vectorstore:    
    st.write(f"주제: '{st.session_state.topic}' 로 Vectorstore 준비완료")
    
    
# 2. 사용자 질문에 유사한 내용을 Vectorstore에서 RAG 기반으로 답변
user_query = st.text_input('질문을 입력하세요.')

if st.button('질문하기') and user_query:
    # 사용자의 질문을 히스토리에 추가
    st.session_state.chat_history.append(f"[user]: {user_query}")
    st.text(f'[You]: {user_query}')    

    # response = st.session_state.model.generate_content(user_querie)
    # model_response = response.candidates[0].content.parts[0].text
        
    # 모델 응답 RAG
    if st.session_state.vectorstore:    
        response, context_docs = rag_chatbot(user_query)        
        st.text(f'[Chatbot]: {response}')
        st.text(f'출처 문서:\n')        
        st.write(context_docs)
    else: 
        response = "vector store is not ready."
        st.text(f'[Chatbot]: {response}')
    
    # 모델 응답을 히스토리에 추가
    st.session_state.chat_history.append(f"[chatbot]: {response}")
    
    # 전체 히스토리 출력
    st.text("Chat History")
    st.text('--------------------------------------------')
    st.text("\n".join(st.session_state.chat_history))
