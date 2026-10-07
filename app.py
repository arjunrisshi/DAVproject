import os
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
import uvicorn
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sqlalchemy import Column
from sqlmodel import SQLModel, Field as SQLField, JSON, create_engine, Session, select, func

# ==============================================================================
# 1. DATABASE SETUP
# ==============================================================================
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./edm_framework.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, echo=False, connect_args=connect_args)

def get_session():
    with Session(engine) as session:
        yield session

# ==============================================================================
# 2. SQLMODEL SCHEMA
# ==============================================================================
class Student(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = SQLField(default=None, primary_key=True)
    student_id: str = SQLField(unique=True)
    name: str
    email: str
    major: str
    cohort_year: int
    
    attendance_rate: float
    lms_engagement_score: float
    assignment_completion: float
    midterm_score: float
    forum_participation: int
    quiz_average: float
    
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    shap_explanation: Optional[dict] = SQLField(default=None, sa_column=Column(JSON))
    last_evaluated: Optional[datetime] = SQLField(default_factory=lambda: datetime.now(timezone.utc))

# ==============================================================================
# 3. PYDANTIC SCHEMAS
# ==============================================================================
class StudentCreate(BaseModel):
    student_id: str
    name: str
    email: str
    major: str
    cohort_year: int = 2026
    attendance_rate: float = Field(..., ge=0, le=100)
    lms_engagement_score: float = Field(..., ge=0, le=100)
    assignment_completion: float = Field(..., ge=0, le=100)
    midterm_score: float = Field(..., ge=0, le=100)
    forum_participation: int = Field(..., ge=0)
    quiz_average: float = Field(..., ge=0, le=100)

class SHAPContribution(BaseModel):
    feature: str
    impact: float
    description: str

class RiskPredictionOutput(BaseModel):
    student_id: str
    risk_score: float
    risk_level: str
    base_value: float
    top_factors: List[SHAPContribution]

class RecommendationOutput(BaseModel):
    student_id: str
    risk_level: str
    primary_risk_driver: str
    action_plan: List[str]
    suggested_resources: List[str]

class ModelMetric(BaseModel):
    model_name: str
    category: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

# ==============================================================================
# 4. HYBRID MODELS & BENCHMARK ENGINE
# ==============================================================================
FEATURE_COLUMNS = [
    "attendance_rate",
    "lms_engagement_score",
    "assignment_completion",
    "midterm_score",
    "forum_participation",
    "quiz_average"
]

class FeatureLevelHybridPCA_RF:
    def __init__(self):
        self.pca = PCA(n_components=4)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, X, y):
        X_pca = self.pca.fit_transform(X)
        self.rf.fit(X_pca, y)
        return self

    def predict(self, X):
        return self.rf.predict(self.pca.transform(X))

    def predict_proba(self, X):
        return self.rf.predict_proba(self.pca.transform(X))

class StatisticalMLHybrid:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss")

    def fit(self, X, y):
        X_stat = X.copy()
        X_stat['trend_mean'] = X.mean(axis=1)
        X_stat['trend_std'] = X.std(axis=1)
        self.model.fit(X_stat, y)
        return self

    def predict(self, X):
        X_stat = X.copy()
        X_stat['trend_mean'] = X.mean(axis=1)
        X_stat['trend_std'] = X.std(axis=1)
        return self.model.predict(X_stat)

    def predict_proba(self, X):
        X_stat = X.copy()
        X_stat['trend_mean'] = X.mean(axis=1)
        X_stat['trend_std'] = X.std(axis=1)
        return self.model.predict_proba(X_stat)

class DeepLearningProxyHybrid:
    def __init__(self):
        self.dense_feats = RandomDenseExtractor()
        self.rf = RandomForestClassifier(n_estimators=120, random_state=42)

    def fit(self, X, y):
        X_emb = self.dense_feats.transform(X)
        self.rf.fit(X_emb, y)
        return self

    def predict(self, X):
        return self.rf.predict(self.dense_feats.transform(X))

    def predict_proba(self, X):
        return self.rf.predict_proba(self.dense_feats.transform(X))

class RandomDenseExtractor:
    def __init__(self):
        np.random.seed(42)
        self.weights = np.random.normal(0, 0.1, (6, 12))

    def transform(self, X):
        return np.maximum(0, np.dot(X, self.weights))

class ProposedRFE_XGBoostHybrid:
    def __init__(self, n_features_to_select=5):
        self.rfe = RFE(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_features_to_select=n_features_to_select
        )
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.03,
            subsample=0.8, random_state=42, eval_metric="logloss"
        )
        
    def fit(self, X_train, y_train):
        X_rfe = self.rfe.fit_transform(X_train, y_train)
        self.xgb_model.fit(X_rfe, y_train)
        return self

    def predict(self, X_test):
        X_rfe = self.rfe.transform(X_test)
        return self.xgb_model.predict(X_rfe)

    def predict_proba(self, X_test):
        X_rfe = self.rfe.transform(X_test)
        return self.xgb_model.predict_proba(X_rfe)

class DAVHybridXAIEngine:
    def __init__(self):
        self.comparison_metrics = []
        self.proposed_hybrid_model = None
        self.explainer = None
        self._train_and_evaluate_all_experiments()

    def _train_and_evaluate_all_experiments(self):
        np.random.seed(42)
        n_samples = 2000
        
        attendance = np.random.uniform(30, 100, n_samples)
        lms_engagement = np.random.uniform(10, 100, n_samples)
        assignments = np.random.uniform(20, 100, n_samples)
        midterm = np.random.uniform(20, 100, n_samples)
        forum = np.random.poisson(lam=3, size=n_samples)
        quizzes = np.random.uniform(25, 100, n_samples)
        
        X = pd.DataFrame({
            "attendance_rate": attendance,
            "lms_engagement_score": lms_engagement,
            "assignment_completion": assignments,
            "midterm_score": midterm,
            "forum_participation": forum,
            "quiz_average": quizzes
        })
        
        risk_logit = (
            - 0.05 * attendance
            - 0.04 * lms_engagement
            - 0.06 * assignments
            - 0.03 * midterm
            - 0.22 * forum
            - 0.03 * quizzes
            + 11.5
        )
        probs = 1 / (1 + np.exp(-risk_logit))
        y = (probs > 0.5).astype(int)

        split_idx = int(n_samples * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        experiments = [
            ("Logistic Regression", "Baseline Model", LogisticRegression()),
            ("Decision Tree", "Baseline Model", DecisionTreeClassifier(max_depth=5, random_state=42)),
            ("Random Forest", "Existing Ensemble", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("Standard XGBoost", "Existing Boosting", xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")),
            ("A. PCA + Random Forest", "Feature-level Hybrid", FeatureLevelHybridPCA_RF()),
            ("B. Stacking Ensemble (LR+RF+XGB)", "Algorithm-level Hybrid", StackingClassifier(
                estimators=[('lr', LogisticRegression()), ('rf', RandomForestClassifier(n_estimators=50, random_state=42))],
                final_estimator=xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss")
            )),
            ("C. Statistical Trend + ML", "Statistical + ML Hybrid", StatisticalMLHybrid()),
            ("D. Neural Embeddings + Random Forest", "DL + ML Hybrid", DeepLearningProxyHybrid()),
        ]

        for name, category, model in experiments:
            model.fit(X_train, y_train)
            self._evaluate_model(name, category, model, X_test, y_test)

        self.proposed_hybrid_model = ProposedRFE_XGBoostHybrid(n_features_to_select=5)
        self.proposed_hybrid_model.fit(X_train, y_train)
        self._evaluate_model("E. Proposed Hybrid (RFE + XGBoost)", "Optimization + ML", self.proposed_hybrid_model, X_test, y_test)

        self.explainer = shap.TreeExplainer(self.proposed_hybrid_model.xgb_model)

    def _evaluate_model(self, name: str, category: str, model: Any, X_test: pd.DataFrame, y_test: np.ndarray):
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
        
        self.comparison_metrics.append({
            "model_name": name,
            "category": category,
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4)
        })

    def predict_and_explain(self, feature_dict: Dict[str, Any]):
        df = pd.DataFrame([feature_dict])[FEATURE_COLUMNS]
        
        risk_score = float(self.proposed_hybrid_model.predict_proba(df)[0][1])
        risk_score = round(risk_score, 2)
        
        if risk_score >= 0.50:
            risk_level = "High"
        elif risk_score >= 0.25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
            
        df_rfe = self.proposed_hybrid_model.rfe.transform(df)
        selected_indices = np.where(self.proposed_hybrid_model.rfe.support_)[0]
        selected_features = [FEATURE_COLUMNS[i] for i in selected_indices]

        shap_vals = self.explainer(df_rfe)
        base_val = float(shap_vals.base_values[0])
        shap_array = shap_vals.values[0]

        factors = []
        for idx, name in enumerate(selected_features):
            impact = float(shap_array[idx])
            direction = "Elevates Risk" if impact > 0 else "Reduces Risk"

            factors.append({
                "feature": name,
                "impact": round(impact, 4),
                "description": f"{name.replace('_', ' ').title()} ({direction})"
            })
            
        factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return risk_score, risk_level, round(base_val, 3), factors

ml_engine = DAVHybridXAIEngine()

# ==============================================================================
# 5. DATA SEEDING & APP LIFECYCLE
# ==============================================================================
def seed_data_if_empty():
    with Session(engine) as session:
        existing = session.exec(select(func.count(Student.id))).one()
        if existing == 0:
            student_profiles = [
                {"id": "STU-1001", "name": "Rahul Sharma", "major": "Computer Science", "att": 35.0, "lms": 28.0, "assign": 30.0, "mid": 38.0, "forum": 0, "quiz": 40.0},
                {"id": "STU-1002", "name": "Ananya Verma", "major": "Data Science", "att": 40.0, "lms": 32.0, "assign": 35.0, "mid": 42.0, "forum": 1, "quiz": 45.0},
                {"id": "STU-1003", "name": "Vikram Reddy", "major": "Computer Science", "att": 42.0, "lms": 30.0, "assign": 38.0, "mid": 35.0, "forum": 0, "quiz": 38.0},
                {"id": "STU-1008", "name": "Priya Nair", "major": "Data Science", "att": 68.0, "lms": 65.0, "assign": 65.0, "mid": 60.0, "forum": 3, "quiz": 64.0},
                {"id": "STU-1009", "name": "Rohan Gupta", "major": "Computer Science", "att": 65.0, "lms": 58.0, "assign": 62.0, "mid": 58.0, "forum": 2, "quiz": 60.0},
                {"id": "STU-1015", "name": "Arjun Risshi", "major": "Computer Science", "att": 95.0, "lms": 92.0, "assign": 98.0, "mid": 94.0, "forum": 12, "quiz": 96.0},
            ]

            for p in student_profiles:
                feat_dict = {
                    "attendance_rate": p["att"],
                    "lms_engagement_score": p["lms"],
                    "assignment_completion": p["assign"],
                    "midterm_score": p["mid"],
                    "forum_participation": p["forum"],
                    "quiz_average": p["quiz"],
                }
                risk_score, risk_level, base_val, factors = ml_engine.predict_and_explain(feat_dict)
                student = Student(
                    student_id=p["id"],
                    name=p["name"],
                    email=f"{p['id'].lower()}@university.edu",
                    major=p["major"],
                    cohort_year=2026,
                    risk_score=risk_score,
                    risk_level=risk_level,
                    shap_explanation={"base_value": base_val, "top_factors": factors},
                    **feat_dict
                )
                session.add(student)
            session.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    SQLModel.metadata.create_all(engine)
    seed_data_if_empty()
    yield

app = FastAPI(title="Explainable AI EDM Framework API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==============================================================================
# 6. REST ENDPOINTS
# ==============================================================================
@app.get("/", response_class=HTMLResponse)
def read_root():
    if os.path.exists("index.html"):
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>EDM Framework API Running. Ensure index.html exists.</h1>", status_code=200)

@app.post("/api/v1/students/", response_model=Student)
def create_student(student_in: StudentCreate, session: Session = Depends(get_session)):
    existing = session.exec(select(Student).where(Student.student_id == student_in.student_id)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Student ID already exists.")

    feat_dict = {
        "attendance_rate": float(student_in.attendance_rate),
        "lms_engagement_score": float(student_in.lms_engagement_score),
        "assignment_completion": float(student_in.assignment_completion),
        "midterm_score": float(student_in.midterm_score),
        "forum_participation": int(student_in.forum_participation),
        "quiz_average": float(student_in.quiz_average)
    }
    
    risk_score, risk_level, base_val, factors = ml_engine.predict_and_explain(feat_dict)

    db_student = Student(
        student_id=student_in.student_id,
        name=student_in.name,
        email=student_in.email,
        major=student_in.major,
        cohort_year=student_in.cohort_year,
        attendance_rate=student_in.attendance_rate,
        lms_engagement_score=student_in.lms_engagement_score,
        assignment_completion=student_in.assignment_completion,
        midterm_score=student_in.midterm_score,
        forum_participation=student_in.forum_participation,
        quiz_average=student_in.quiz_average,
        risk_score=risk_score,
        risk_level=risk_level,
        shap_explanation={"base_value": base_val, "top_factors": factors}
    )

    try:
        session.add(db_student)
        session.commit()
        session.refresh(db_student)
        return db_student
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"Database insertion failed: {str(e)}")

@app.get("/api/v1/students/", response_model=List[Student])
def list_students(risk_level: Optional[str] = Query(None), limit: int = 50, session: Session = Depends(get_session)):
    query = select(Student)
    if risk_level:
        query = query.where(Student.risk_level == risk_level)
    return session.exec(query.limit(limit)).all()

@app.get("/api/v1/students/{student_id}/xai-explain", response_model=RiskPredictionOutput)
def get_student_explanation(student_id: str, session: Session = Depends(get_session)):
    student = session.exec(select(Student).where(Student.student_id == student_id)).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    feat_dict = {col: getattr(student, col) for col in FEATURE_COLUMNS}
    risk_score, risk_level, base_val, factors = ml_engine.predict_and_explain(feat_dict)

    return RiskPredictionOutput(
        student_id=student.student_id,
        risk_score=risk_score,
        risk_level=risk_level,
        base_value=base_val,
        top_factors=factors
    )

@app.get("/api/v1/analytics/model-comparison", response_model=List[ModelMetric])
def get_model_comparison_table():
    return ml_engine.comparison_metrics

@app.get("/api/v1/students/{student_id}/recommendations", response_model=RecommendationOutput)
def get_adaptive_interventions(student_id: str, session: Session = Depends(get_session)):
    student = session.exec(select(Student).where(Student.student_id == student_id)).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    feat_dict = {col: getattr(student, col) for col in FEATURE_COLUMNS}
    _, risk_level, _, factors = ml_engine.predict_and_explain(feat_dict)
    
    risk_increasing_factors = [f for f in factors if f["impact"] > 0]
    top_factor = risk_increasing_factors[0]["feature"] if risk_increasing_factors else factors[0]["feature"]
    
    if top_factor == "attendance_rate":
        action_plans = ["Schedule 1-on-1 counselor check-in", "Enable daily SMS attendance notifications"]
        resources = ["Academic Advising Hub", "Flexible Absence Policy Guide"]
    elif top_factor == "assignment_completion":
        action_plans = ["Assign peer mentor for milestone tracking", "Break final project into weekly micro-tasks"]
        resources = ["Writing & Homework Lab", "Time Management Workshop"]
    elif top_factor == "lms_engagement_score":
        action_plans = ["Flag for interactive digital modules", "Trigger LMS push reminders for quizzes"]
        resources = ["Digital Learning Portal", "Interactive Quiz Refreshes"]
    else:
        action_plans = ["Recommend targeted tutoring for midterm recovery", "Invite to weekly study group"]
        resources = ["Subject Specific Tutoring", "Exam Prep Repositories"]

    return RecommendationOutput(
        student_id=student.student_id,
        risk_level=risk_level,
        primary_risk_driver=top_factor.replace("_", " ").title(),
        action_plan=action_plans,
        suggested_resources=resources
    )

@app.get("/api/v1/analytics/summary")
def get_cohort_summary(session: Session = Depends(get_session)):
    total_students = session.exec(select(func.count(Student.id))).one()
    high_risk = session.exec(select(func.count(Student.id)).where(Student.risk_level == "High")).one()
    medium_risk = session.exec(select(func.count(Student.id)).where(Student.risk_level == "Medium")).one()
    low_risk = session.exec(select(func.count(Student.id)).where(Student.risk_level == "Low")).one()

    return {
        "total_students": total_students,
        "risk_distribution": {
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk
        }
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)