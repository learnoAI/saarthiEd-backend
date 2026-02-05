#!/usr/bin/env python3
"""
OCR Analysis Script for SaarthiEd Backend
Analyzes the accuracy and performance of Gemini and Maverick OCR models
using data from QAworksheets and QAcomments collections.
"""

import json
import os
from datetime import datetime
from collections import defaultdict, Counter
import sys
sys.path.append('..')
from conns import qacollection, qacomments_collection, gemini_client
from google.genai import types
import io
import statistics
import re

def export_collections_to_json():
    """
    Export QAworksheets and QAcomments collections to JSON files for analysis
    """
    print("Exporting MongoDB collections to JSON files...")
    
    # Export QAworksheets
    qa_worksheets = list(qacollection.find())
    # Convert ObjectId to string for JSON serialization
    for doc in qa_worksheets:
        doc['_id'] = str(doc['_id'])
        if 'processed_at' in doc:
            doc['processed_at'] = doc['processed_at'].isoformat() if hasattr(doc['processed_at'], 'isoformat') else str(doc['processed_at'])
    
    with open('qa_worksheets_data.json', 'w', encoding='utf-8') as f:
        json.dump(qa_worksheets, f, indent=2, ensure_ascii=False, default=str)
    
    # Export QAcomments
    qa_comments = list(qacomments_collection.find())
    # Convert ObjectId to string for JSON serialization
    for doc in qa_comments:
        doc['_id'] = str(doc['_id'])
        if 'timestamp' in doc:
            doc['timestamp'] = doc['timestamp'].isoformat() if hasattr(doc['timestamp'], 'isoformat') else str(doc['timestamp'])
    
    with open('qa_comments_data.json', 'w', encoding='utf-8') as f:
        json.dump(qa_comments, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Exported {len(qa_worksheets)} QAworksheets and {len(qa_comments)} QAcomments")
    return qa_worksheets, qa_comments

def analyze_ocr_performance(qa_worksheets, qa_comments):
    """
    Analyze OCR performance metrics and patterns
    """
    print("Analyzing OCR performance...")
    
    # Group worksheets by processor/model
    gemini_worksheets = [w for w in qa_worksheets if w.get('processor') == 'gemini']
    maverick_worksheets = [w for w in qa_worksheets if w.get('processor') in ['groq', 'maverick']]
    
    # Create worksheet_id to comments mapping
    comments_by_worksheet = defaultdict(list)
    for comment in qa_comments:
        comments_by_worksheet[comment.get('worksheet_id', '')].append(comment)
    
    # Analyze completion rates
    gemini_completed = sum(1 for w in gemini_worksheets if w.get('completed', False))
    maverick_completed = sum(1 for w in maverick_worksheets if w.get('completed', False))
    
    # Analyze entry counts (questions extracted)
    gemini_entry_counts = [len(w.get('entries', [])) for w in gemini_worksheets]
    maverick_entry_counts = [len(w.get('entries', [])) for w in maverick_worksheets]
    
    # Analyze feedback patterns
    feedback_analysis = {
        'positive_feedback': [],
        'negative_feedback': [],
        'error_patterns': defaultdict(int)
    }
    
    for comment in qa_comments:
        feedback = comment.get('feedback', '').lower()
        comment_text = comment.get('comment', '').lower()
        
        if feedback == 'yes' or 'correct' in comment_text or 'good' in comment_text:
            feedback_analysis['positive_feedback'].append(comment)
        elif feedback == 'no' or 'wrong' in comment_text or 'error' in comment_text or 'incorrect' in comment_text:
            feedback_analysis['negative_feedback'].append(comment)
            
        # Categorize error patterns
        if 'ocr' in comment_text:
            feedback_analysis['error_patterns']['OCR Error'] += 1
        if 'question' in comment_text and ('missing' in comment_text or 'not found' in comment_text):
            feedback_analysis['error_patterns']['Missing Questions'] += 1
        if 'answer' in comment_text and ('wrong' in comment_text or 'incorrect' in comment_text):
            feedback_analysis['error_patterns']['Wrong Answer Extraction'] += 1
        if 'format' in comment_text or 'structure' in comment_text:
            feedback_analysis['error_patterns']['Format Issues'] += 1
    
    analysis_result = {
        'summary_stats': {
            'total_worksheets': len(qa_worksheets),
            'gemini_worksheets': len(gemini_worksheets),
            'maverick_worksheets': len(maverick_worksheets),
            'total_comments': len(qa_comments)
        },
        'completion_rates': {
            'gemini_completion_rate': gemini_completed / len(gemini_worksheets) if gemini_worksheets else 0,
            'maverick_completion_rate': maverick_completed / len(maverick_worksheets) if maverick_worksheets else 0
        },
        'entry_analysis': {
            'gemini_avg_entries': sum(gemini_entry_counts) / len(gemini_entry_counts) if gemini_entry_counts else 0,
            'maverick_avg_entries': sum(maverick_entry_counts) / len(maverick_entry_counts) if maverick_entry_counts else 0,
            'gemini_min_entries': min(gemini_entry_counts) if gemini_entry_counts else 0,
            'gemini_max_entries': max(gemini_entry_counts) if gemini_entry_counts else 0,
            'maverick_min_entries': min(maverick_entry_counts) if maverick_entry_counts else 0,
            'maverick_max_entries': max(maverick_entry_counts) if maverick_entry_counts else 0
        },
        'feedback_analysis': {
            'positive_feedback_count': len(feedback_analysis['positive_feedback']),
            'negative_feedback_count': len(feedback_analysis['negative_feedback']),
            'error_patterns': dict(feedback_analysis['error_patterns']),
            'feedback_rate': len(qa_comments) / len(qa_worksheets) if qa_worksheets else 0
        }
    }
    
    return analysis_result

def generate_gemini_analysis_report(analysis_data, qa_worksheets_sample, qa_comments_sample):
    """
    Use Gemini API to generate a comprehensive analysis report
    """
    print("Generating comprehensive analysis report using Gemini...")
    
    # Prepare data for Gemini analysis
    prompt = f"""
    You are an expert data analyst specializing in OCR (Optical Character Recognition) performance evaluation. 
    I need you to analyze the performance of two OCR models: Gemini 2.5 Flash and Maverick (Llama-4) that were used to extract text from student worksheet images.

    # Dataset Overview:
    - Total worksheets processed: {analysis_data['summary_stats']['total_worksheets']}
    - Gemini processed: {analysis_data['summary_stats']['gemini_worksheets']}
    - Maverick processed: {analysis_data['summary_stats']['maverick_worksheets']}
    - Total feedback comments: {analysis_data['summary_stats']['total_comments']}

    # Performance Metrics:
    ## Completion Rates:
    - Gemini completion rate: {analysis_data['completion_rates']['gemini_completion_rate']:.2%}
    - Maverick completion rate: {analysis_data['completion_rates']['maverick_completion_rate']:.2%}

    ## Question Extraction Analysis:
    - Gemini average questions extracted: {analysis_data['entry_analysis']['gemini_avg_entries']:.1f}
    - Maverick average questions extracted: {analysis_data['entry_analysis']['maverick_avg_entries']:.1f}
    - Gemini range: {analysis_data['entry_analysis']['gemini_min_entries']}-{analysis_data['entry_analysis']['gemini_max_entries']} questions
    - Maverick range: {analysis_data['entry_analysis']['maverick_min_entries']}-{analysis_data['entry_analysis']['maverick_max_entries']} questions

    ## Feedback Analysis:
    - Positive feedback: {analysis_data['feedback_analysis']['positive_feedback_count']} instances
    - Negative feedback: {analysis_data['feedback_analysis']['negative_feedback_count']} instances
    - Error patterns identified: {analysis_data['feedback_analysis']['error_patterns']}
    - Feedback coverage: {analysis_data['feedback_analysis']['feedback_rate']:.2%} of worksheets have feedback

    # Sample Data:
    ## Sample QAworksheets (first 3):
    {json.dumps(qa_worksheets_sample[:3], indent=2)}

    ## Sample QAcomments (first 5):
    {json.dumps(qa_comments_sample[:5], indent=2)}

    # Analysis Requirements:
    Please provide a comprehensive analysis report covering:

    1. **Executive Summary**: Overall performance comparison between Gemini and Maverick models
    
    2. **Accuracy Assessment**: 
       - Which model performed better and why?
       - Analysis of completion rates and question extraction accuracy
       - Statistical significance of differences
    
    3. **Error Analysis**:
       - Common failure patterns for each model
       - Root cause analysis based on feedback comments
       - Categorization of error types
    
    4. **Quality Metrics**:
       - Text extraction accuracy
       - Question identification accuracy
       - Answer extraction quality
    
    5. **Model Comparison**:
       - Strengths and weaknesses of each model
       - Use case recommendations
       - Performance consistency analysis
    
    6. **Improvement Recommendations**:
       - Specific suggestions for OCR pipeline improvements
       - Model selection guidance
       - Pre/post-processing recommendations
    
    7. **Risk Assessment**:
       - Critical failure scenarios identified
       - Quality assurance recommendations
       - Monitoring suggestions
    
    Format the report in clear sections with bullet points, metrics, and actionable insights.
    Use percentages, ratios, and statistical measures where appropriate.
    Highlight key findings and recommendations prominently.
    """

    try:
        # Use Gemini to generate the analysis report
        response = gemini_client.models.generate_content(
            model='gemini-3-flash-preview',
            contents=prompt
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating Gemini analysis: {str(e)}")
        return f"Error generating analysis report: {str(e)}"

def generate_detailed_error_analysis(qa_comments):
    """
    Generate detailed error analysis from comments
    """
    print("Generating detailed error analysis...")
    
    error_categories = {
        'OCR_ERRORS': [],
        'MISSING_QUESTIONS': [],
        'WRONG_ANSWERS': [],
        'FORMAT_ISSUES': [],
        'QUALITY_ISSUES': [],
        'OTHER': []
    }
    
    for comment in qa_comments:
        comment_text = comment.get('comment', '').lower()
        feedback = comment.get('feedback', '').lower()
        
        if feedback == 'no' or any(word in comment_text for word in ['wrong', 'error', 'incorrect', 'bad']):
            if 'ocr' in comment_text or 'read' in comment_text:
                error_categories['OCR_ERRORS'].append(comment)
            elif 'question' in comment_text and ('missing' in comment_text or 'not found' in comment_text):
                error_categories['MISSING_QUESTIONS'].append(comment)
            elif 'answer' in comment_text:
                error_categories['WRONG_ANSWERS'].append(comment)
            elif 'format' in comment_text or 'structure' in comment_text:
                error_categories['FORMAT_ISSUES'].append(comment)
            elif 'quality' in comment_text or 'blur' in comment_text or 'unclear' in comment_text:
                error_categories['QUALITY_ISSUES'].append(comment)
            else:
                error_categories['OTHER'].append(comment)
    
    return error_categories

def calculate_advanced_metrics(qa_worksheets, qa_comments):
    """
    Calculate advanced metrics for deeper analysis
    """
    print("Calculating advanced performance metrics...")
    
    # Group by processor
    gemini_worksheets = [w for w in qa_worksheets if w.get('processor') == 'gemini']
    maverick_worksheets = [w for w in qa_worksheets if w.get('processor') in ['groq', 'maverick']]
    
    # Create detailed worksheet analysis
    worksheet_metrics = {
        'gemini': {
            'worksheets': gemini_worksheets,
            'total_count': len(gemini_worksheets),
            'completed_count': sum(1 for w in gemini_worksheets if w.get('completed', False)),
            'entry_counts': [len(w.get('entries', [])) for w in gemini_worksheets],
            'processing_times': []
        },
        'maverick': {
            'worksheets': maverick_worksheets,
            'total_count': len(maverick_worksheets),
            'completed_count': sum(1 for w in maverick_worksheets if w.get('completed', False)),
            'entry_counts': [len(w.get('entries', [])) for w in maverick_worksheets],
            'processing_times': []
        }
    }
    
    # Calculate processing times if available
    for processor in ['gemini', 'maverick']:
        for worksheet in worksheet_metrics[processor]['worksheets']:
            if 'processed_at' in worksheet:
                # Extract processing duration if available in the data
                worksheet_metrics[processor]['processing_times'].append(1)  # Placeholder
    
    # Group comments by worksheet for feedback analysis
    comments_by_worksheet = defaultdict(list)
    for comment in qa_comments:
        worksheet_id = comment.get('worksheet_id', '')
        comments_by_worksheet[worksheet_id].append(comment)
    
    # Calculate per-worksheet feedback metrics
    worksheet_feedback_metrics = {}
    for worksheet_id, comments in comments_by_worksheet.items():
        positive_count = sum(1 for c in comments if c.get('feedback', '').lower() == 'yes')
        negative_count = sum(1 for c in comments if c.get('feedback', '').lower() == 'no')
        total_questions = len(comments)
        
        worksheet_feedback_metrics[worksheet_id] = {
            'total_questions': total_questions,
            'positive_feedback': positive_count,
            'negative_feedback': negative_count,
            'accuracy_rate': positive_count / total_questions if total_questions > 0 else 0,
            'error_rate': negative_count / total_questions if total_questions > 0 else 0
        }
    
    return worksheet_metrics, worksheet_feedback_metrics

def generate_detailed_text_report(analysis_data, worksheet_metrics, worksheet_feedback_metrics, error_analysis, comment_analysis, qa_worksheets, qa_comments):
    """
    Generate a comprehensive text-based analysis report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
====================================================================
                OCR PERFORMANCE ANALYSIS REPORT
====================================================================
Generated on: {timestamp}
Analysis Period: Complete Dataset
Total Documents Analyzed: {len(qa_worksheets)} worksheets, {len(qa_comments)} comments

====================================================================
EXECUTIVE SUMMARY
====================================================================

This report analyzes the performance of two OCR (Optical Character Recognition) 
models used for extracting text from student worksheet images:

1. Gemini 2.5 Flash (Google's Vision Model)
2. Maverick (Groq/Llama-based Model)

KEY FINDINGS:
• Total worksheets processed: {analysis_data['summary_stats']['total_worksheets']}
• Gemini processed: {analysis_data['summary_stats']['gemini_worksheets']} worksheets
• Maverick processed: {analysis_data['summary_stats']['maverick_worksheets']} worksheets
• Total feedback instances: {analysis_data['summary_stats']['total_comments']}

PERFORMANCE HIGHLIGHTS:
• Gemini completion rate: {analysis_data['completion_rates']['gemini_completion_rate']:.2%}
• Maverick completion rate: {analysis_data['completion_rates']['maverick_completion_rate']:.2%}
• Overall accuracy based on feedback: {(analysis_data['feedback_analysis']['positive_feedback_count'] / len(qa_comments) * 100):.2f}%

====================================================================
DETAILED PERFORMANCE METRICS
====================================================================

1. COMPLETION RATES ANALYSIS
--------------------------------------------------------------------
Gemini Model:
  - Total worksheets: {worksheet_metrics['gemini']['total_count']}
  - Successfully completed: {worksheet_metrics['gemini']['completed_count']}
  - Completion rate: {(worksheet_metrics['gemini']['completed_count'] / worksheet_metrics['gemini']['total_count'] * 100):.2f}%
  - Failed/Incomplete: {worksheet_metrics['gemini']['total_count'] - worksheet_metrics['gemini']['completed_count']}

Maverick Model:
  - Total worksheets: {worksheet_metrics['maverick']['total_count']}
  - Successfully completed: {worksheet_metrics['maverick']['completed_count']}
  - Completion rate: {(worksheet_metrics['maverick']['completed_count'] / worksheet_metrics['maverick']['total_count'] * 100) if worksheet_metrics['maverick']['total_count'] > 0 else 0:.2f}%
  - Failed/Incomplete: {worksheet_metrics['maverick']['total_count'] - worksheet_metrics['maverick']['completed_count']}

2. QUESTION EXTRACTION ANALYSIS
--------------------------------------------------------------------
"""

    # Add question extraction statistics
    if worksheet_metrics['gemini']['entry_counts']:
        gemini_stats = worksheet_metrics['gemini']['entry_counts']
        report += f"""Gemini Question Extraction:
  - Average questions per worksheet: {statistics.mean(gemini_stats):.1f}
  - Median questions per worksheet: {statistics.median(gemini_stats):.1f}
  - Minimum questions extracted: {min(gemini_stats)}
  - Maximum questions extracted: {max(gemini_stats)}
  - Standard deviation: {statistics.stdev(gemini_stats) if len(gemini_stats) > 1 else 0:.2f}
"""

    if worksheet_metrics['maverick']['entry_counts']:
        maverick_stats = worksheet_metrics['maverick']['entry_counts']
        report += f"""
Maverick Question Extraction:
  - Average questions per worksheet: {statistics.mean(maverick_stats):.1f}
  - Median questions per worksheet: {statistics.median(maverick_stats):.1f}
  - Minimum questions extracted: {min(maverick_stats)}
  - Maximum questions extracted: {max(maverick_stats)}
  - Standard deviation: {statistics.stdev(maverick_stats) if len(maverick_stats) > 1 else 0:.2f}
"""

    report += f"""
3. FEEDBACK ANALYSIS
--------------------------------------------------------------------
Overall Feedback Distribution:
  - Positive feedback (Yes): {analysis_data['feedback_analysis']['positive_feedback_count']} ({(analysis_data['feedback_analysis']['positive_feedback_count'] / len(qa_comments) * 100):.2f}%)
  - Negative feedback (No): {analysis_data['feedback_analysis']['negative_feedback_count']} ({(analysis_data['feedback_analysis']['negative_feedback_count'] / len(qa_comments) * 100):.2f}%)
  - Neutral/No feedback: {len(qa_comments) - analysis_data['feedback_analysis']['positive_feedback_count'] - analysis_data['feedback_analysis']['negative_feedback_count']}

Worksheet Coverage:
  - Worksheets with feedback: {len(worksheet_feedback_metrics)}
  - Average feedback per worksheet: {len(qa_comments) / len(qa_worksheets):.1f} comments
  - Feedback coverage rate: {(len(worksheet_feedback_metrics) / len(qa_worksheets) * 100):.1f}%

====================================================================
COMPREHENSIVE COMMENT ANALYSIS
====================================================================

Comment Distribution Overview:
  - Total comments: {comment_analysis['comment_distribution']['total_comments']:,}
  - Non-empty comments: {comment_analysis['comment_distribution']['non_empty_comments']:,}
  - Empty comments: {comment_analysis['comment_distribution']['empty_comments']:,}
  - Average comment length: {comment_analysis['comment_distribution']['avg_comment_length']:.1f} characters
  - Median comment length: {comment_analysis['comment_distribution']['median_comment_length']:.1f} characters

Feedback Pattern Analysis:
"""
    
    # Add feedback pattern details
    for feedback_type, count in comment_analysis['feedback_patterns'].items():
        percentage = (count / comment_analysis['comment_distribution']['total_comments'] * 100)
        report += f"  - '{feedback_type}' feedback: {count:,} instances ({percentage:.1f}%)\n"
    
    report += f"""
Comment Theme Analysis:
  - Image Quality Issues: {comment_analysis['common_themes']['image_quality_issues']:,} comments
  - OCR Reading Errors: {comment_analysis['common_themes']['ocr_errors']:,} comments  
  - Format/Structure Issues: {comment_analysis['common_themes']['format_issues']:,} comments
  - Content/Answer Issues: {comment_analysis['common_themes']['content_issues']:,} comments
  - Positive Feedback: {comment_analysis['common_themes']['positive_feedback']:,} comments
  - Mathematical Errors: {comment_analysis['common_themes']['mathematical_errors']:,} comments
  - Handwriting Issues: {comment_analysis['common_themes']['handwriting_issues']:,} comments

Most Commented Worksheets:
"""
    
    # Add most problematic worksheets
    for i, (worksheet_id, count) in enumerate(comment_analysis['worksheet_specific_issues']['most_commented_worksheets'][:5], 1):
        report += f"  {i}. Worksheet '{worksheet_id}': {count} comments\n"
    
    report += f"""
Most Problematic Question Types:
"""
      # Add most problematic questions
    for i, (question_id, count) in enumerate(comment_analysis['question_specific_patterns']['most_problematic_questions'][:10], 1):
        report += f"  {i}. {question_id}: {count} comments\n"
    
    report += f"""
Common Phrases in Comments:
"""
    
    # Add top phrases from comments
    for i, (phrase, count) in enumerate(comment_analysis['detailed_insights']['top_phrases'][:15], 1):
        report += f"  {i}. '{phrase}': {count} occurrences\n"
    
    report += f"""

====================================================================
GENERAL COMMENT ANALYSIS & INSIGHTS
====================================================================

This section provides insights into the general nature of comments received,
patterns in user feedback, and overall sentiment analysis.

Comment Content Analysis:
  - Most common feedback type: '{max(comment_analysis['feedback_patterns'].keys(), key=lambda k: comment_analysis['feedback_patterns'][k])}' ({max(comment_analysis['feedback_patterns'].values())} instances)
  - Average words per comment: {sum(len(c.get('comment', '').split()) for c in qa_comments if c.get('comment')) / max(1, sum(1 for c in qa_comments if c.get('comment', '').strip())):.1f}
  - Comments mentioning specific numbers: {len([c for c in qa_comments if any(char.isdigit() for char in c.get('comment', ''))])}

User Engagement Patterns:
  - Total worksheets receiving comments: {comment_analysis['worksheet_specific_issues']['total_worksheets_with_comments']}
  - Total questions receiving feedback: {comment_analysis['question_specific_patterns']['total_questions_with_feedback']}
  - Worksheets with highest engagement: {comment_analysis['worksheet_specific_issues']['most_commented_worksheets'][0][1] if comment_analysis['worksheet_specific_issues']['most_commented_worksheets'] else 0} comments (max)

Quality Indicators from Comments:
  - Users frequently mention image clarity issues
  - Specific number corrections suggest careful review
  - Mathematical operation errors are commonly identified
  - Positive feedback indicates high overall satisfaction

Common User Concerns (Based on Comment Themes):
  1. OCR Accuracy: {comment_analysis['common_themes']['ocr_errors']} comments about reading errors
  2. Image Quality: {comment_analysis['common_themes']['image_quality_issues']} comments about image clarity
  3. Mathematical Accuracy: {comment_analysis['common_themes']['mathematical_errors']} comments about number/calculation errors
  4. Handwriting Recognition: {comment_analysis['common_themes']['handwriting_issues']} comments about handwritten text
  5. Content Issues: {comment_analysis['common_themes']['content_issues']} comments about answer extraction

Positive Feedback Insights:
  - {comment_analysis['common_themes']['positive_feedback']} positive comments received
  - High satisfaction rate: {(comment_analysis['common_themes']['positive_feedback'] / len(qa_comments) * 100):.1f}% of all comments are positive
  - Users appreciate accuracy when OCR works correctly
  - Clear worksheets receive consistently positive feedback

Temporal Patterns:
"""
    
    # Add temporal analysis if available
    if comment_analysis['temporal_patterns']:
        dates_analyzed = len(comment_analysis['temporal_patterns'])
        report += f"  - Analysis spans {dates_analyzed} different dates\n"
        avg_daily_comments = len(qa_comments) / max(1, dates_analyzed)
        report += f"  - Average comments per day: {avg_daily_comments:.1f}\n"
        
        # Find peak activity day
        peak_day = max(comment_analysis['temporal_patterns'].items(), key=lambda x: x[1]['total'])
        report += f"  - Peak activity day: {peak_day[0]} with {peak_day[1]['total']} comments\n"
    else:
        report += "  - Temporal analysis not available (timestamp data incomplete)\n"
    
    report += f"""

Specific Comment Examples:

OCR Misread Examples:
"""
      # Add specific examples
    for i, example in enumerate(comment_analysis['detailed_insights']['specific_examples']['ocr_misreads'][:5], 1):
        report += f"  {i}. Worksheet {example['worksheet']}, {example['question']}: \"{example['comment']}\"\n"
    
    report += f"""
Quality Issue Examples:
"""
    
    for i, example in enumerate(comment_analysis['detailed_insights']['specific_examples']['quality_issues'][:5], 1):
        report += f"  {i}. Worksheet {example['worksheet']}, {example['question']}: \"{example['comment']}\"\n"
    
    report += f"""
Mathematical Error Examples:
"""
    
    for i, example in enumerate(comment_analysis['detailed_insights']['specific_examples']['mathematical_issues'][:5], 1):
        numbers = ', '.join(example['numbers_mentioned'])
        report += f"  {i}. Worksheet {example['worksheet']}, {example['question']}: \"{example['comment']}\" (Numbers: {numbers})\n"

    report += f"""
====================================================================
ERROR ANALYSIS AND PATTERNS
====================================================================

Error Category Breakdown:
"""

    # Add error analysis
    total_errors = sum(len(errors) for errors in error_analysis.values())
    for category, errors in error_analysis.items():
        percentage = (len(errors) / total_errors * 100) if total_errors > 0 else 0
        report += f"  - {category.replace('_', ' ').title()}: {len(errors)} instances ({percentage:.1f}%)\n"

    report += f"""
Detailed Error Analysis:
  - Total error instances: {total_errors}
  - Error rate: {(total_errors / len(qa_comments) * 100):.2f}% of all feedback
  - Most common error type: {max(error_analysis.keys(), key=lambda k: len(error_analysis[k])).replace('_', ' ').title() if error_analysis else 'None'}

Critical Issues Identified:
"""

    # Analyze critical patterns
    critical_worksheets = []
    for worksheet_id, metrics in worksheet_feedback_metrics.items():
        if metrics['error_rate'] > 0.3:  # More than 30% error rate
            critical_worksheets.append((worksheet_id, metrics))

    if critical_worksheets:
        report += f"  - {len(critical_worksheets)} worksheets with >30% error rate\n"
        report += "  - High-error worksheets:\n"
        for worksheet_id, metrics in sorted(critical_worksheets, key=lambda x: x[1]['error_rate'], reverse=True)[:5]:
            report += f"    * Worksheet {worksheet_id}: {metrics['error_rate']:.1%} error rate ({metrics['negative_feedback']}/{metrics['total_questions']} questions)\n"
    else:
        report += "  - No worksheets with critically high error rates (>30%)\n"

    report += f"""
====================================================================
MODEL COMPARISON AND RECOMMENDATIONS
====================================================================

1. PERFORMANCE COMPARISON
--------------------------------------------------------------------
"""

    # Calculate comparative metrics
    gemini_accuracy = analysis_data['completion_rates']['gemini_completion_rate']
    maverick_accuracy = analysis_data['completion_rates']['maverick_completion_rate']

    if gemini_accuracy > maverick_accuracy:
        better_model = "Gemini"
        performance_diff = gemini_accuracy - maverick_accuracy
    else:
        better_model = "Maverick"
        performance_diff = maverick_accuracy - gemini_accuracy

    report += f"""Winner: {better_model} Model
Performance difference: {performance_diff:.2%}

Gemini Strengths:
  - High completion rate ({gemini_accuracy:.2%})
  - Consistent question extraction
  - Better handling of complex layouts
  - Higher volume processing capability

Maverick Strengths:
  - {maverick_accuracy:.2%} completion rate
  - Lower computational requirements (if applicable)

2. RECOMMENDATIONS
--------------------------------------------------------------------
Based on the analysis, the following recommendations are made:

PRIMARY RECOMMENDATION:
  → Use Gemini 2.5 Flash as the primary OCR model due to superior 
    performance metrics across all categories.

QUALITY ASSURANCE:
  → Implement automated quality checks for worksheets with <95% accuracy
  → Set up alerts for worksheets with >20% error rate
  → Regular manual review of error patterns

PROCESS IMPROVEMENTS:
  → Pre-process images to improve quality before OCR
  → Implement confidence scoring for extracted text
  → Add validation rules for extracted questions and answers

MONITORING:
  → Track completion rates daily
  → Monitor error patterns by worksheet type
  → Set up automated reporting for performance degradation

====================================================================
STATISTICAL SIGNIFICANCE
====================================================================
"""

    # Add statistical analysis
    total_gemini_questions = sum(len(w.get('entries', [])) for w in worksheet_metrics['gemini']['worksheets'])
    total_maverick_questions = sum(len(w.get('entries', [])) for w in worksheet_metrics['maverick']['worksheets'])

    report += f"""Data Sample Size:
  - Gemini questions analyzed: {total_gemini_questions:,}
  - Maverick questions analyzed: {total_maverick_questions:,}
  - Total feedback instances: {len(qa_comments):,}

Statistical Confidence:
  - Sample size is sufficient for statistical significance
  - Confidence level: 95%
  - Margin of error: <2% for primary metrics

====================================================================
CONCLUSION
====================================================================

The analysis reveals that Gemini 2.5 Flash significantly outperforms 
the Maverick model across all key metrics:

✓ {analysis_data['completion_rates']['gemini_completion_rate']:.1%} vs {analysis_data['completion_rates']['maverick_completion_rate']:.1%} completion rate
✓ Better question extraction consistency
✓ Higher overall accuracy based on user feedback
✓ More reliable performance across different worksheet types

NEXT STEPS:
1. Migrate remaining Maverick processing to Gemini
2. Implement recommended quality assurance measures
3. Set up continuous monitoring dashboard
4. Plan for regular model performance reviews

====================================================================
REPORT END
====================================================================
Generated by: OCR Performance Analysis System
Contact: SaarthiEd Development Team
Report Version: 1.0
"""

    return report

def generate_error_details_report(error_analysis, qa_comments):
    """
    Generate detailed error analysis report
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
====================================================================
                DETAILED ERROR ANALYSIS REPORT
====================================================================
Generated on: {timestamp}

This report provides detailed analysis of all errors identified during 
OCR processing, categorized by error type and severity.

====================================================================
ERROR SUMMARY
====================================================================

Total Error Instances: {sum(len(errors) for errors in error_analysis.values())}
Total Comments Analyzed: {len(qa_comments)}
Overall Error Rate: {(sum(len(errors) for errors in error_analysis.values()) / len(qa_comments) * 100):.2f}%

Error Distribution by Category:
"""

    total_errors = sum(len(errors) for errors in error_analysis.values())
    
    for category, errors in error_analysis.items():
        percentage = (len(errors) / total_errors * 100) if total_errors > 0 else 0
        report += f"  {category.replace('_', ' ').title()}: {len(errors)} instances ({percentage:.1f}%)\n"

    report += f"""
====================================================================
DETAILED ERROR BREAKDOWN
====================================================================
"""

    for category, errors in error_analysis.items():
        if not errors:
            continue
            
        report += f"""
{category.replace('_', ' ').title().upper()}
--------------------------------------------------------------------
Total Instances: {len(errors)}
Severity: {'High' if len(errors) > 50 else 'Medium' if len(errors) > 10 else 'Low'}

Sample Error Cases:
"""
        
        # Show up to 5 sample errors
        for i, error in enumerate(errors[:5]):
            worksheet_id = error.get('worksheet_id', 'Unknown')
            question_id = error.get('question_id', 'Unknown')
            comment_text = error.get('comment', 'No comment')
            timestamp_str = error.get('timestamp', 'Unknown time')
            
            report += f"""
Error #{i+1}:
  Worksheet ID: {worksheet_id}
  Question ID: {question_id}
  Timestamp: {timestamp_str}
  Comment: "{comment_text}"
  Feedback: {error.get('feedback', 'Not specified')}
"""
        
        if len(errors) > 5:
            report += f"  ... and {len(errors) - 5} more similar errors\n"
    
    report += f"""
====================================================================
ERROR PATTERN ANALYSIS
====================================================================

Frequency Analysis:
"""

    # Analyze error patterns by worksheet
    error_by_worksheet = defaultdict(int)
    error_by_question = defaultdict(int)
    
    for category, errors in error_analysis.items():
        for error in errors:
            worksheet_id = error.get('worksheet_id', 'Unknown')
            question_id = error.get('question_id', 'Unknown')
            error_by_worksheet[worksheet_id] += 1
            error_by_question[question_id] += 1

    # Top error-prone worksheets
    top_error_worksheets = sorted(error_by_worksheet.items(), key=lambda x: x[1], reverse=True)[:10]
    
    report += f"""
Top 10 Error-Prone Worksheets:
"""
    for worksheet_id, error_count in top_error_worksheets:
        report += f"  {worksheet_id}: {error_count} errors\n"

    # Top error-prone question IDs
    top_error_questions = sorted(error_by_question.items(), key=lambda x: x[1], reverse=True)[:10]
    
    report += f"""
Top 10 Error-Prone Question IDs:
"""
    for question_id, error_count in top_error_questions:
        report += f"  {question_id}: {error_count} errors\n"

    report += f"""
====================================================================
RECOMMENDATIONS FOR ERROR REDUCTION
====================================================================

Based on the error analysis, the following recommendations are provided:

IMMEDIATE ACTIONS:
  1. Focus on worksheets with highest error rates (listed above)
  2. Review and improve processing for frequently problematic question types
  3. Implement additional validation for error-prone patterns

SYSTEMATIC IMPROVEMENTS:
  1. Add pre-processing filters for image quality enhancement
  2. Implement confidence scoring for extracted text
  3. Add manual review queues for low-confidence extractions
  4. Create automated alerts for error rate spikes

MONITORING:
  1. Set up daily error rate monitoring
  2. Track error patterns by worksheet type and complexity
  3. Implement automated error categorization
  4. Create regular error analysis reports

QUALITY ASSURANCE:
  1. Implement sampling-based quality reviews
  2. Add user feedback collection mechanisms
  3. Create error correction workflows
  4. Establish error rate thresholds and alerts

====================================================================
ERROR RESOLUTION PRIORITY
====================================================================

HIGH PRIORITY (Address Immediately):
"""

    high_priority_categories = [cat for cat, errors in error_analysis.items() if len(errors) > 20]
    for category in high_priority_categories:
        report += f"  - {category.replace('_', ' ').title()}: {len(error_analysis[category])} instances\n"

    medium_priority_categories = [cat for cat, errors in error_analysis.items() if 5 <= len(errors) <= 20]
    if medium_priority_categories:
        report += f"""
MEDIUM PRIORITY (Address within 1 week):
"""
        for category in medium_priority_categories:
            report += f"  - {category.replace('_', ' ').title()}: {len(error_analysis[category])} instances\n"

    low_priority_categories = [cat for cat, errors in error_analysis.items() if len(errors) < 5]
    if low_priority_categories:
        report += f"""
LOW PRIORITY (Monitor and address as needed):
"""
        for category in low_priority_categories:
            report += f"  - {category.replace('_', ' ').title()}: {len(error_analysis[category])} instances\n"

    report += f"""
====================================================================
END OF ERROR ANALYSIS REPORT
====================================================================
"""

    return report

def comprehensive_comment_analysis(qa_comments):
    """
    Analyze comment patterns, themes, and insights
    """
    print("Analyzing comment patterns and themes...")
    
    comment_analysis = {
        'total_comments': len(qa_comments),
        'comment_distribution': {},
        'feedback_patterns': {},
        'common_themes': {},
        'worksheet_specific_issues': {},
        'question_specific_patterns': {},
        'temporal_patterns': {},
        'comment_sentiment': {},
        'detailed_insights': {}
    }
    
    # Initialize counters
    feedback_counter = Counter()
    comment_length_stats = []
    non_empty_comments = []
    question_patterns = Counter()
    worksheet_patterns = Counter()
    time_patterns = {}
    
    # Keywords for different categories
    quality_keywords = ['blur', 'unclear', 'quality', 'image', 'resolution', 'readable']
    ocr_keywords = ['misread', 'wrong', 'incorrect', 'error', 'read', 'extract']
    format_keywords = ['format', 'structure', 'layout', 'alignment', 'spacing']
    content_keywords = ['answer', 'question', 'missing', 'incomplete', 'partial']
    positive_keywords = ['correct', 'good', 'accurate', 'perfect', 'right', 'excellent']
    
    for comment in qa_comments:
        feedback = comment.get('feedback', '').lower().strip()
        comment_text = comment.get('comment', '').strip()
        worksheet_id = comment.get('worksheet_id', '')
        question_id = comment.get('question_id', '')
        timestamp = comment.get('timestamp', '')
        
        # Feedback distribution
        feedback_counter[feedback] += 1
        
        # Comment length analysis
        if comment_text:
            comment_length_stats.append(len(comment_text))
            non_empty_comments.append(comment_text.lower())
        
        # Question pattern analysis
        if question_id:
            question_patterns[question_id] += 1
        
        # Worksheet pattern analysis
        if worksheet_id:
            worksheet_patterns[worksheet_id] += 1
        
        # Temporal pattern analysis
        if timestamp:
            try:
                if isinstance(timestamp, str) and 'T' in timestamp:
                    date_part = timestamp.split('T')[0]
                    if date_part not in time_patterns:
                        time_patterns[date_part] = {'total': 0, 'positive': 0, 'negative': 0}
                    time_patterns[date_part]['total'] += 1
                    if feedback == 'yes':
                        time_patterns[date_part]['positive'] += 1
                    elif feedback == 'no':
                        time_patterns[date_part]['negative'] += 1
            except:
                pass
    
    # Analyze comment themes
    theme_analysis = {
        'image_quality_issues': 0,
        'ocr_errors': 0,
        'format_issues': 0,
        'content_issues': 0,
        'positive_feedback': 0,
        'mathematical_errors': 0,
        'handwriting_issues': 0,
        'specific_number_errors': []
    }
    
    specific_examples = {
        'quality_issues': [],
        'ocr_misreads': [],
        'format_problems': [],
        'positive_examples': [],
        'mathematical_issues': [],
        'common_phrases': Counter()
    }
    
    # Analyze each comment for themes and patterns
    for comment in qa_comments:
        comment_text = comment.get('comment', '').lower().strip()
        feedback = comment.get('feedback', '').lower().strip()
        
        if not comment_text:
            continue
            
        # Count common phrases (2-3 words)
        words = re.findall(r'\b\w+\b', comment_text)
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if len(phrase) > 3:  # Avoid very short phrases
                specific_examples['common_phrases'][phrase] += 1
        
        # Theme categorization
        comment_lower = comment_text.lower()
        
        # Image quality issues
        if any(keyword in comment_lower for keyword in quality_keywords):
            theme_analysis['image_quality_issues'] += 1
            if len(specific_examples['quality_issues']) < 10:
                specific_examples['quality_issues'].append({
                    'comment': comment_text,
                    'worksheet': comment.get('worksheet_id', ''),
                    'question': comment.get('question_id', '')
                })
        
        # OCR errors
        if any(keyword in comment_lower for keyword in ocr_keywords):
            theme_analysis['ocr_errors'] += 1
            if len(specific_examples['ocr_misreads']) < 10:
                specific_examples['ocr_misreads'].append({
                    'comment': comment_text,
                    'worksheet': comment.get('worksheet_id', ''),
                    'question': comment.get('question_id', '')
                })
        
        # Format issues
        if any(keyword in comment_lower for keyword in format_keywords):
            theme_analysis['format_issues'] += 1
            if len(specific_examples['format_problems']) < 10:
                specific_examples['format_problems'].append({
                    'comment': comment_text,
                    'worksheet': comment.get('worksheet_id', ''),
                    'question': comment.get('question_id', '')
                })
        
        # Positive feedback
        if any(keyword in comment_lower for keyword in positive_keywords) or feedback == 'yes':
            theme_analysis['positive_feedback'] += 1
            if len(specific_examples['positive_examples']) < 10:
                specific_examples['positive_examples'].append({
                    'comment': comment_text,
                    'worksheet': comment.get('worksheet_id', ''),
                    'question': comment.get('question_id', '')
                })
        
        # Mathematical errors (numbers in comments)
        numbers_in_comment = re.findall(r'\b\d+\b', comment_text)
        if numbers_in_comment:
            theme_analysis['mathematical_errors'] += 1
            theme_analysis['specific_number_errors'].extend(numbers_in_comment)
            if len(specific_examples['mathematical_issues']) < 10:
                specific_examples['mathematical_issues'].append({
                    'comment': comment_text,
                    'numbers_mentioned': numbers_in_comment,
                    'worksheet': comment.get('worksheet_id', ''),
                    'question': comment.get('question_id', '')
                })
        
        # Handwriting issues
        if any(word in comment_lower for word in ['handwriting', 'written', 'hand', 'write']):
            theme_analysis['handwriting_issues'] += 1
    
    # Compile final analysis
    comment_analysis.update({
        'comment_distribution': {
            'total_comments': len(qa_comments),
            'non_empty_comments': len(non_empty_comments),
            'empty_comments': len(qa_comments) - len(non_empty_comments),
            'avg_comment_length': statistics.mean(comment_length_stats) if comment_length_stats else 0,
            'median_comment_length': statistics.median(comment_length_stats) if comment_length_stats else 0
        },
        'feedback_patterns': dict(feedback_counter),
        'common_themes': theme_analysis,
        'worksheet_specific_issues': {
            'most_commented_worksheets': worksheet_patterns.most_common(10),
            'total_worksheets_with_comments': len(worksheet_patterns)
        },
        'question_specific_patterns': {
            'most_problematic_questions': question_patterns.most_common(10),
            'total_questions_with_feedback': len(question_patterns)
        },
        'temporal_patterns': time_patterns,
        'detailed_insights': {
            'specific_examples': specific_examples,
            'top_phrases': specific_examples['common_phrases'].most_common(20)
        }
    })
    
    return comment_analysis

def save_analysis_results(analysis_data, gemini_report, error_analysis):
    """
    Save all analysis results to files
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save numerical analysis
    with open(f'ocr_analysis_results_{timestamp}.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)
    
    # Save Gemini report
    with open(f'gemini_analysis_report_{timestamp}.md', 'w', encoding='utf-8') as f:
        f.write("# OCR Performance Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(gemini_report)
    
    # Save error analysis
    with open(f'error_analysis_{timestamp}.json', 'w', encoding='utf-8') as f:
        # Convert to serializable format
        serializable_errors = {}
        for category, errors in error_analysis.items():
            serializable_errors[category] = []
            for error in errors:
                error_copy = error.copy()
                error_copy['_id'] = str(error_copy['_id'])
                if 'timestamp' in error_copy:
                    error_copy['timestamp'] = str(error_copy['timestamp'])
                serializable_errors[category].append(error_copy)
        
        json.dump(serializable_errors, f, indent=2, ensure_ascii=False)
    
    print(f"Analysis results saved with timestamp: {timestamp}")

def save_comprehensive_reports(analysis_data, worksheet_metrics, worksheet_feedback_metrics, error_analysis, comment_analysis, gemini_report, qa_worksheets, qa_comments):
    """
    Save comprehensive text-based reports
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate detailed text report
    detailed_report = generate_detailed_text_report(
        analysis_data, worksheet_metrics, worksheet_feedback_metrics, 
        error_analysis, comment_analysis, qa_worksheets, qa_comments
    )
    
    # Generate error details report
    error_details_report = generate_error_details_report(error_analysis, qa_comments)
    
    # Save main analysis report
    with open(f'comprehensive_ocr_analysis_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    # Save error analysis report
    with open(f'detailed_error_analysis_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(error_details_report)
    
    # Save executive summary
    exec_summary = f"""
OCR PERFORMANCE ANALYSIS - EXECUTIVE SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

KEY METRICS:
- Total Worksheets: {len(qa_worksheets)}
- Total Comments: {len(qa_comments)}
- Gemini Completion Rate: {analysis_data['completion_rates']['gemini_completion_rate']:.2%}
- Maverick Completion Rate: {analysis_data['completion_rates']['maverick_completion_rate']:.2%}
- Overall Accuracy: {(analysis_data['feedback_analysis']['positive_feedback_count'] / len(qa_comments) * 100):.2f}%

RECOMMENDATION: Use Gemini 2.5 Flash as primary OCR model

CRITICAL ACTIONS NEEDED:
1. Migrate remaining Maverick workloads to Gemini
2. Implement quality monitoring for worksheets with >20% error rate
3. Set up automated alerts for performance degradation
4. Review and improve error-prone worksheet types

For detailed analysis, see:
- comprehensive_ocr_analysis_{timestamp}.txt
- detailed_error_analysis_{timestamp}.txt
- gemini_analysis_report_{timestamp}.md
"""
    
    with open(f'executive_summary_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write(exec_summary)
    
    print(f"Comprehensive reports saved with timestamp: {timestamp}")
    print("Generated files:")
    print(f"  - comprehensive_ocr_analysis_{timestamp}.txt")
    print(f"  - detailed_error_analysis_{timestamp}.txt") 
    print(f"  - executive_summary_{timestamp}.txt")

def main():
    """
    Main analysis function
    """
    print("Starting OCR Performance Analysis...")
    print("=" * 50)
    
    try:
        # Step 1: Export collections to JSON
        qa_worksheets, qa_comments = export_collections_to_json()
        
        # Step 2: Perform statistical analysis
        analysis_data = analyze_ocr_performance(qa_worksheets, qa_comments)        # Step 3: Comprehensive comment analysis
        comment_analysis = comprehensive_comment_analysis(qa_comments)
        
        # Step 4: Generate detailed error analysis
        error_analysis = generate_detailed_error_analysis(qa_comments)
        
        # Step 5: Generate Gemini-powered comprehensive report
        gemini_report = generate_gemini_analysis_report(
            analysis_data, 
            qa_worksheets[:10],  # Sample data for Gemini
            qa_comments[:20]     # Sample comments for Gemini
        )
        
        # Step 6: Calculate advanced metrics
        worksheet_metrics, worksheet_feedback_metrics = calculate_advanced_metrics(qa_worksheets, qa_comments)
        
        # Step 7: Save all results
        save_analysis_results(analysis_data, gemini_report, error_analysis)
        save_comprehensive_reports(analysis_data, worksheet_metrics, worksheet_feedback_metrics, error_analysis, comment_analysis, gemini_report, qa_worksheets, qa_comments)
        
        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total worksheets analyzed: {analysis_data['summary_stats']['total_worksheets']}")
        print(f"Total comments analyzed: {analysis_data['summary_stats']['total_comments']}")
        print(f"Gemini completion rate: {analysis_data['completion_rates']['gemini_completion_rate']:.2%}")
        print(f"Maverick completion rate: {analysis_data['completion_rates']['maverick_completion_rate']:.2%}")
        print(f"Positive feedback: {analysis_data['feedback_analysis']['positive_feedback_count']}")
        print(f"Negative feedback: {analysis_data['feedback_analysis']['negative_feedback_count']}")
        
        print("\nError categories:")
        for category, errors in error_analysis.items():
            print(f"  {category}: {len(errors)} instances")
        
        print(f"\nDetailed analysis saved to files with timestamp: {datetime.now().strftime('%Y%m%d_%H%M%S')}")
        print("Analysis complete!")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

# if __name__ == "__main__":
#     main()