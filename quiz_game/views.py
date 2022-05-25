from django.db.models.query import QuerySet
from django.shortcuts import render
from quiz_game.models import Player, Question
from quiz_game.forms import UploadFileForm
from django.http import HttpResponseRedirect
import pandas as pd
import random

# Create your views here.
def title(request):
    return render(request, 'title.html', {})


def home(request):
    player = Player.objects.filter(name="サンプルユーザー").first()
    name = player.name
    athretic = player.athretic
    bmi = player.BMI
    sleep = player.sleep
    knowledge = player.knowledge
    total = athretic + bmi + sleep + knowledge

    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            print("ファイルを受け取りました")
            file = request.FILES['file']
            df_questions = pd.read_csv(file, encoding="shift-jis").dropna(how='all')
            for df_question in df_questions.iterrows():
                question = df_question[1]["問題文"]
                stored_question = Question.objects.get(question=question)
                print(stored_question)
                if not stored_question:
                    category = df_question[1]["カテゴリ"]
                    choice1 = df_question[1]["選択肢1"]
                    choice2 = df_question[1]["選択肢2"]
                    choice3 = df_question[1]["選択肢3"]
                    choice4 = df_question[1]["選択肢4"]
                    answer = df_question[1]["解答"]
                    explanation = df_question[1]["解説"]
                    reference = df_question[1]["参考文献"]
                    question_obj = Question(
                        question = question,
                        category = category,
                        answer = answer,
                        choice1 = choice1,
                        choice2 = choice2,
                        choice3 = choice3,
                        choice4 = choice4,
                        explanation = explanation,
                        reference = reference)
                    question_obj.save()
    form = UploadFileForm()

    return render(request, 'home.html', {
        "name": name,
        "athretic": athretic,
        "bmi": bmi,
        "sleep": sleep,
        "knowledge": knowledge,
        "total": total,
        "form": form
    })


def add_questions(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            pass
        return HttpResponseRedirect('/home')
    else:
        form = UploadFileForm()
    return HttpResponseRedirect('/home')


def select_stage(request):
    return render(request, 'select_stage.html', {})


def battle_disease(request):
    question_list = []
    questions = Question.objects.all()
    # questions = Question.objects.filter(category="病気の問題")
    for question_obj in questions:
        question_data = {
            "question": question_obj.question,
            "choice_1": question_obj.choice1,
            "choice_2": question_obj.choice2,
            "choice_3": question_obj.choice3,
            "choice_4": question_obj.choice4,
            "answer": question_obj.answer,
            "explanation": question_obj.explanation,
            "reference": question_obj.reference,
        }
        question_list.append(question_data)
    random.shuffle(question_list)
    print(question_list)
    return render(request, 'battle_disease.html', {
        "question_list": question_list
        })