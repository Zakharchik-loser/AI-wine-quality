import base64
import io
import pandas as pd
import shap
from django.contrib import messages
from django.contrib.auth import authenticate,login
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import render, redirect
import matplotlib

import sys
import os

from utils.ai_explain import generate_rag

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from llm.nyahh import search
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
from .forms import CustomUserCreationForm
from .train_model import y_test, y_pred

# Create your views here.

model = joblib.load(r"C:\Users\zakha\PycharmProjects\Some_sort_of_ml/model/config_model/HotChicken.pkl")

def home(request):
    return HttpResponse("Welcome to the new project!")



def register(request):

    if request.method == "POST":
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect('login')
        else:
            messages.error(request,"Parasha")

            return render(request, "register.html", {"form": form})
    else:
        form = CustomUserCreationForm()



        return render(request,"register.html",{"form":form})



def login_user(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request,username=username,password=password)
        if user is not None:
            login(request,user)
            return render(request,"main.html")
        else:
            messages.error(request,"You're not existing bruh")

            return render(request,"login.html")


    return render(request,"login.html")

@login_required
def main_page(request):
    return render(request, "main.html")


def predict_quality(request):
    if request.method == "POST":
        alcohol = float(request.POST.get("alcohol"))
        pH = float(request.POST.get("pH"))
        fixed_acidity = float(request.POST.get("fixed_acidity"))
        volatile_acidity = float(request.POST.get("volatile_acidity"))


        X = np.array([[
            fixed_acidity,
            volatile_acidity,
            0.0,  # citric acid
            0.0,  # residual sugar
            0.0,  # chlorides
            0.0,  # free sulfur dioxide
            0.0,  # total sulfur dioxide
            0.0,  # density
            pH,
            0.0,  # sulphates
            alcohol
        ]])

        prediction = model.predict(X)[0]

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X)[0]

        shap_list = shap_values.tolist()

        yt = np.array(y_test)
        yp = np.array(y_pred)

        # Sort by true values to get a smooth line
        sorted_idx = np.argsort(yt)
        yt_sorted = yt[sorted_idx]
        yp_sorted = yp[sorted_idx]

        pred_smooth = pd.Series(yp_sorted).rolling(10).mean()
        true_smooth = pd.Series(yt_sorted).rolling(10).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(true_smooth, label="True (smooth)", linewidth=3)
        ax.plot(pred_smooth, label="Predicted (smooth)", linewidth=3)
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)

        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        buffer.close()

        return render(request, "prediction.html", {
            "prediction": round(float(prediction), 2),
            "shap":shap_list,
            "plot":image_base64

        })

    return render(request, "prediction.html")


def rag_usage(request):
    results_list = []
    error = None
    query = ""
    ai_answer = None

    if request.method == "POST":
        query = request.POST.get("query", "").strip()
        if not query:
            error = "Please,enter the request"
        else:
            try:
                raw_results = search(query)

                documents = raw_results.get("documents")
                metadatas = raw_results.get("metadatas")


                if not documents:
                    documents = []
                if not metadatas:
                    metadatas = []


                documents = documents[0] if len(documents) > 0 else []
                metadatas = metadatas[0] if len(metadatas) > 0 else []

                results_list = []
                for doc, meta in zip(documents, metadatas):
                    results_list.append({
                        "text": doc,
                        **(meta or {})
                    })

                if not results_list:
                     error = "Unfortunately nothing was found :("


                else:
                    text_only = [r["text"] for r in results_list]
                    ai_answer = generate_rag(query,text_only)

            except Exception as e:
                error = f"Error while searching: {str(e)}"

    context = {
        "results": results_list,
        "error": error,
        "query": query,
        "ai_answer": ai_answer,
    }
    results_list.sort(key=lambda x: float(x.get("price", 0)), reverse=True)

    return render(request, "rag.html", context)
