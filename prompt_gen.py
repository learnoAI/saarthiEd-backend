import os
import pathlib
import google.generativeai as genai
from PIL import Image
from google.api_core import exceptions as google_exceptions
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set the environment variable and try again.")
    exit()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    print(f"ERROR: Failed to configure Gemini API: {e}")
    exit()

INPUT_DIR = pathlib.Path("context/context_worksheets")
OUTPUT_DIR = pathlib.Path("context/prompts")

META_PROMPT = """
You are an expert educational prompt engineer. Your task is to analyze an image of a student's worksheet and generate a highly specific, tailored JSON-extraction prompt for another AI.

**Your Analysis Process:**
1.  **Identify Core Task:** Determine the fundamental task of the worksheet. Is it "What comes before/after"? Addition? Spelling? Matching? Mention the title if visible.
2.  **Analyze Structure:** Determine the total number of questions, the layout (e.g., two columns), and the numbering scheme (e.g., Q1, Q2...).
3.  **Determine Question Format:** Identify the exact format of a single question. How is the blank space represented? (e.g., `___ , 32`, `5 + 7 = ___`).
4.  **Extract a Concrete Example:** Find one clear, representative question and the student's handwritten answer from the image to use as a non-ambiguous example in the prompt you generate.

**Your Output:**
Based on your analysis, generate a prompt that follows the rules below. Your output must be ONLY the text of the generated prompt, ready to be saved to a file. Do not include any of your own analysis or phrases like "Here is the prompt:".

--- START OF PROMPT TO GENERATE ---
This is a worksheet titled '{worksheet_title}'. The task for the student is to {worksheet_task}. Extract all questions and their corresponding student answers.

<Rules>
1.  **Transcribe Student's Answer Literally:** Your single most important job is to read the student's handwritten answer and transcribe it exactly. DO NOT calculate the correct answer. If the student makes a mathematical error, you must record that error.
    *   **Specific Example from this sheet:** For question {example_question_number}, `{example_question_format}`, the student wrote '{example_student_answer}'. Your output for the `answer` MUST be `'{example_student_answer}'`.

2.  **Strict Question Formatting:** The format for every question on this sheet is `{question_format_description}`. Represent the question field using `___` (three underscores) for the blank.
    *   **Example:** For a question like `{example_question_number}`, the `question` field should be `"{example_question_format}"`.

3.  **Worksheet Structure and Order:** This worksheet has exactly {total_questions} questions, arranged in {layout_description}. You must return all questions in strict numerical order.

4.  **Handwriting Interpretation:** The answers are handwritten by a child. Interpret the digits/letters based on what is physically written. Do not guess the student's intent.

5.  **Unanswered Blanks:** If a student left a blank empty, use an empty string `""` for the `answer` field.
</Rules>

Respond in the following JSON format:
{format_instructions}
--- END OF PROMPT TO GENERATE ---
"""

def generate_prompts():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # *** MAJOR FIX: Corrected the model name ***
    model = genai.GenerativeModel('gemini-2.5-pro')
    print("Gemini model initialized. Starting to process worksheets...")

    if not any(INPUT_DIR.iterdir()):
        print(f"Warning: Input directory '{INPUT_DIR}' is empty. Nothing to process.")
        return

    for image_path in INPUT_DIR.iterdir():
        if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        print(f"\nProcessing: {image_path.name}")
        try:
            image = Image.open(image_path)
            response = model.generate_content([META_PROMPT, image])
            generated_prompt_text = response.text

            # Clean up potential markdown formatting
            generated_prompt_text = generated_prompt_text.strip().removeprefix("```json").removeprefix("```text").strip()

            base_name = image_path.stem
            output_path = OUTPUT_DIR / f"{base_name}.txt"

            output_path.write_text(generated_prompt_text, encoding='utf-8')
            print(f"  -> Successfully generated and saved prompt to: {output_path}")

        # Catch specific Google API errors for better debugging
        except google_exceptions.PermissionDenied as e:
            print(f"  -> ERROR: Permission Denied. This is an API key or API activation issue.")
            print(f"     Details: {e.message}")
            print("     Please verify your API key and ensure the 'Generative Language API' is enabled in your Google Cloud project.")
            break # Stop the script on authentication failure
        except Exception as e:
            print(f"  -> An unexpected ERROR occurred processing {image_path.name}: {e}")

    print("\nProcessing complete.")

if __name__ == "__main__":
    generate_prompts()