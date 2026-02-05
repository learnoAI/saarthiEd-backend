"""
Error Logs Analysis Script
Analyzes all documents in the error_logs collection and saves categorized results to JSON.
"""

import json
import re
from datetime import datetime, timezone
from collections import Counter, defaultdict
from conns import error_logs_collection


def extract_exception_type(stack_trace):
    """Extract exception type from stack trace string."""
    if not stack_trace:
        return None

    # Match patterns like "TypeError:", "ValueError:", etc.
    match = re.search(r'(\w+Error|\w+Exception):', stack_trace)
    if match:
        return match.group(1)
    return None


def analyze_error_logs():
    """Fetch and analyze all error logs from MongoDB."""

    print("Fetching error logs from MongoDB...")

    # Fetch all error documents
    errors = list(error_logs_collection.find({}))
    total_errors = len(errors)

    if total_errors == 0:
        print("No error logs found in the collection.")
        return None

    print(f"Found {total_errors} error logs. Analyzing...")

    # Group errors by type
    error_types_data = defaultdict(lambda: {
        'errors': [],
        'messages': set(),
        'exceptions': []
    })

    # Process each error
    for error in errors:
        error_type = error.get('error_type', 'UNKNOWN')
        error_message = error.get('error_message', '')
        payload = error.get('payload', {})
        stack_trace = error.get('stack_trace', '')
        timestamp = error.get('timestamp', '')

        # Store error details
        error_types_data[error_type]['errors'].append({
            'error_message': error_message,
            'payload': payload,
            'timestamp': timestamp
        })

        # Collect unique messages
        if error_message:
            error_types_data[error_type]['messages'].add(error_message)

        # Extract exception type from stack trace
        exception_type = extract_exception_type(stack_trace)
        if exception_type:
            error_types_data[error_type]['exceptions'].append(exception_type)

    # Build the analysis result
    analysis = {
        'total_errors': total_errors,
        'analyzed_at': datetime.now(timezone.utc).isoformat(),
        'error_types': {}
    }

    # Process each error type
    for error_type, data in error_types_data.items():
        count = len(data['errors'])
        percentage = (count / total_errors) * 100

        # Get top 3 sample errors
        sample_errors = data['errors'][:3]

        # Count exception types
        exception_counter = Counter(data['exceptions'])
        common_exceptions = [
            {'type': exc_type, 'count': exc_count}
            for exc_type, exc_count in exception_counter.most_common(5)
        ]

        analysis['error_types'][error_type] = {
            'count': count,
            'percentage': round(percentage, 2),
            'unique_messages': sorted(list(data['messages'])),
            'unique_message_count': len(data['messages']),
            'sample_errors': sample_errors,
            'common_exceptions': common_exceptions
        }

    return analysis


def save_to_json(analysis, filename='error_logs_analysis.json'):
    """Save analysis results to JSON file."""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis saved to: {filename}")


def print_summary(analysis):
    """Print summary statistics to console."""
    print("\n" + "="*60)
    print("ERROR LOGS ANALYSIS SUMMARY")
    print("="*60)

    print(f"\nTotal Errors: {analysis['total_errors']}")
    print(f"Analyzed At: {analysis['analyzed_at']}")
    print(f"Unique Error Types: {len(analysis['error_types'])}")

    print("\n" + "-"*60)
    print("ERROR TYPE DISTRIBUTION")
    print("-"*60)

    # Sort error types by count (descending)
    sorted_types = sorted(
        analysis['error_types'].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    for error_type, data in sorted_types:
        print(f"\n{error_type}")
        print(f"  Count: {data['count']} ({data['percentage']}%)")
        print(f"  Unique Messages: {data['unique_message_count']}")

        if data['common_exceptions']:
            print(f"  Common Exceptions:")
            for exc in data['common_exceptions']:
                print(f"    - {exc['type']}: {exc['count']} occurrences")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    try:
        # Analyze error logs
        analysis = analyze_error_logs()

        if analysis is None:
            return

        # Print summary to console
        print_summary(analysis)

        # Save to JSON file
        save_to_json(analysis)

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()


# if __name__ == "__main__":
#     main()
