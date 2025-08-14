from typing import List
from pydantic import BaseModel, Field

class QuestionScore(BaseModel):
    question_number: int = Field(..., description="The unique identifier for the question")
    question: str = Field(..., description="The text of the question")
    student_answer: str = Field(..., description="The student's provided answer")
    correct_answer: str = Field(..., description="The expected answer or explanation")
    points_earned: float = Field(..., description="Points earned for this question")
    max_points: float = Field(..., description="Maximum possible points for this question")
    is_correct: bool = Field(..., description="Whether the student's answer is correct")
    feedback: str = Field(..., description="Brief explanation of grading for this question")


class GradingResult(BaseModel):
    total_questions: int = Field(..., description="Total number of questions in the assessment")
    overall_score: float = Field(..., description="Total score obtained out of 40")
    grade_percentage: float = Field(..., description="Percentage score")
    question_scores: List[QuestionScore] = Field(..., description="Detailed scores for each question")
    correct_answers: int = Field(..., description="Number of correct answers")
    wrong_answers: int = Field(..., description="Number of incorrect answers")
    unanswered: int = Field(..., description="Number of unanswered questions")
    overall_feedback: str = Field(..., description="Encouraging feedback for the student in 1 line")
    reason_why: str = Field(..., description="Reason why YOU graded the student this score in 1 line")

class ExtractedQuestion(BaseModel):
    question_number: int = Field(description="The unique identifier for the question")
    question: str = Field(description="The entire text of the question without the student's answer. Do not confuse questions on different columns.")
    student_answer: str = Field(description="The student's answer to the question - the exact answer as written by the student and not necessarily the correct answer")

class ExtractedQuestions(BaseModel):
    questions: List[ExtractedQuestion] = Field(description="The list of questions and their corresponding student answers")
