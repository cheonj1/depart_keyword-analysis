from jinja2 import Environment, FileSystemLoader
import os

def generate_html(context):

    def translate_gender(data):
        if isinstance(data, dict):
            return {k: translate_gender(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [translate_gender(i) for i in data]
        elif isinstance(data, str):
            # 1. 문장 중간에 male/female이 섞여 있어도 싹 다 바꿉니다.
            # lower()를 쓰지 않고 대소문자 모두 대응하기 위해 replace를 연달아 씁니다.
            res = data.replace("female", "여성").replace("Female", "여성")
            res = res.replace("male", "남성").replace("Male", "남성")
            return res
        return data

    # 렌더링 직전에 데이터 싹 훑기
    translated_context = translate_gender(context)

    # --- 기존 로직 ---
    file_loader = FileSystemLoader('templates')
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')

    # 반드시 'translated_context'를 넣어주어야 합니다!
    output = template.render(translated_context)

    output_path = "report.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    return os.path.abspath(output_path)
