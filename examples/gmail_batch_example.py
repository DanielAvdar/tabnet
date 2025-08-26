#!/usr/bin/env python3
"""Gmail API Batch Request Example Script.

This script demonstrates how to use the Gmail API batch utilities
to efficiently fetch emails with proper error handling and best practices.

Requirements:
    pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2

Usage:
    python gmail_batch_example.py --credentials credentials.json --query "from:example@gmail.com"
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

try:
    from pytorch_tabnet.utils import GmailAPIClient, GmailBatchError, batch_operation_with_retry, create_credentials, get_message_headers
except ImportError as e:
    print(f"Error importing Gmail utilities: {e}")
    print("Please ensure pytorch-tabnet is installed with Gmail API dependencies:")
    print("pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2")
    sys.exit(1)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gmail API Batch Request Example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with search query
  python gmail_batch_example.py --credentials creds.json --query "from:notifications@github.com"

  # Fetch recent emails with attachments
  python gmail_batch_example.py --credentials creds.json --query "has:attachment" --days 7

  # Process multiple search queries
  python gmail_batch_example.py --credentials creds.json --batch-search

  # Save detailed results to file
  python gmail_batch_example.py --credentials creds.json --query "is:important" --output results.json
        """,
    )

    parser.add_argument("--credentials", type=Path, required=True, help="Path to Google API credentials JSON file")

    parser.add_argument("--token", type=Path, default="token.json", help="Path to store/load OAuth token (default: token.json)")

    parser.add_argument("--query", type=str, default="", help='Gmail search query (e.g., "from:example@gmail.com")')

    parser.add_argument("--days", type=int, help="Limit search to emails from the last N days")

    parser.add_argument("--max-results", type=int, default=100, help="Maximum number of messages to process (default: 100)")

    parser.add_argument("--batch-size", type=int, default=25, help="Batch size for API requests (default: 25, max: 100)")

    parser.add_argument("--output", type=Path, help="Output file to save results (JSON format)")

    parser.add_argument("--batch-search", action="store_true", help="Demonstrate batch search with predefined queries")

    parser.add_argument(
        "--format",
        choices=["minimal", "full", "metadata", "raw"],
        default="metadata",
        help="Message format to retrieve (default: metadata)",
    )

    parser.add_argument(
        "--headers", nargs="+", default=["From", "Subject", "Date", "To"], help="Headers to extract when using metadata format"
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser.parse_args()


def build_search_query(base_query: str, days: int = None) -> str:
    """Build Gmail search query with optional date filter."""
    query_parts = []

    if base_query:
        query_parts.append(base_query)

    if days:
        date_str = (datetime.now() - timedelta(days=days)).strftime("%Y/%m/%d")
        query_parts.append(f"after:{date_str}")

    return " ".join(query_parts)


def demonstrate_batch_search(gmail_client: GmailAPIClient, max_results: int) -> dict:
    """Demonstrate batch search operations with multiple queries."""
    logging.info("Demonstrating batch search operations...")

    # Define interesting search queries
    search_queries = [
        "from:github.com",
        "from:stackoverflow.com",
        "has:attachment",
        "is:important",
        "in:inbox is:unread",
        "from:notifications",
    ]

    # Add date filter for recent emails
    date_7_days_ago = (datetime.now() - timedelta(days=7)).strftime("%Y/%m/%d")
    search_queries = [f"{query} after:{date_7_days_ago}" for query in search_queries]

    # Perform batch search
    search_results = gmail_client.search_messages_batch(search_queries, max_results_per_query=max_results // len(search_queries))

    # Display results
    total_messages = 0
    for query, messages in search_results.items():
        count = len(messages)
        total_messages += count
        logging.info(f"Query '{query}': {count} messages")

    logging.info(f"Total messages across all searches: {total_messages}")
    return search_results


def process_messages(gmail_client: GmailAPIClient, messages: list, format_type: str, headers: list, batch_size: int) -> list:
    """Process messages using batch operations."""
    if not messages:
        logging.info("No messages to process")
        return []

    logging.info(f"Processing {len(messages)} messages in batches of {batch_size}")

    # Extract message IDs
    message_ids = [msg["id"] for msg in messages]

    # Define batch processing function
    def fetch_batch(ids_batch):
        return gmail_client.fetch_messages_batch(
            message_ids=ids_batch,
            format_type=format_type,
            metadata_headers=headers if format_type == "metadata" else None,
            batch_size=batch_size,
        )

    # Process with retry logic
    try:
        detailed_messages = batch_operation_with_retry(
            fetch_batch,
            message_ids,
            batch_size=batch_size * 4,  # Process multiple API batches at once
            max_retries=3,
            retry_delay=1.0,
        )

        logging.info(f"Successfully fetched details for {len(detailed_messages)} messages")
        return detailed_messages

    except Exception as e:
        logging.error(f"Failed to process messages: {e}")
        return []


def extract_message_info(messages: list, headers: list) -> list:
    """Extract useful information from messages."""
    extracted_info = []

    for message in messages:
        # Get basic message info
        info = {
            "id": message.get("id"),
            "thread_id": message.get("threadId"),
            "labels": message.get("labelIds", []),
            "size_estimate": message.get("sizeEstimate"),
            "snippet": message.get("snippet", "")[:100] + "..." if message.get("snippet") else "",
        }

        # Extract headers
        if headers:
            message_headers = get_message_headers(message, headers)
            info["headers"] = message_headers

        extracted_info.append(info)

    return extracted_info


def generate_summary(processed_messages: list) -> None:
    """Generate and display summary statistics."""
    if not processed_messages:
        logging.info("No messages to summarize")
        return

    from collections import Counter

    logging.info("=== EMAIL SUMMARY ===")
    logging.info(f"Total messages processed: {len(processed_messages)}")

    # Analyze senders
    senders = []
    subjects = []
    labels = []

    for msg in processed_messages:
        headers = msg.get("headers", {})

        # Extract sender domain
        from_header = headers.get("From", "")
        if "@" in from_header:
            domain = from_header.split("@")[-1].split(">")[0].strip()
            senders.append(domain)

        # Extract subject keywords
        subject = headers.get("Subject", "")
        if subject:
            subjects.append(subject)

        # Extract labels
        labels.extend(msg.get("labels", []))

    # Top sender domains
    if senders:
        domain_counts = Counter(senders)
        logging.info("\nTop 5 sender domains:")
        for domain, count in domain_counts.most_common(5):
            logging.info(f"  {domain}: {count} messages")

    # Top labels
    if labels:
        label_counts = Counter(labels)
        logging.info("\nTop 5 labels:")
        for label, count in label_counts.most_common(5):
            logging.info(f"  {label}: {count} messages")

    # Subject analysis
    if subjects:
        logging.info("\nSample subjects:")
        for subject in subjects[:5]:
            logging.info(f"  {subject}")


def save_results(data: any, output_file: Path) -> None:
    """Save results to JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results to {output_file}: {e}")


def main():
    """Main function."""
    args = parse_arguments()
    setup_logging(args.verbose)

    logging.info("Gmail API Batch Request Example Starting...")

    try:
        # Step 1: Setup authentication
        logging.info("Setting up authentication...")
        credentials = create_credentials(str(args.credentials), str(args.token))

        # Step 2: Initialize Gmail client
        gmail_client = GmailAPIClient(credentials)
        logging.info("Gmail API client initialized successfully")

        # Step 3: Choose operation mode
        if args.batch_search:
            # Demonstrate batch search operations
            search_results = demonstrate_batch_search(gmail_client, args.max_results)

            # Collect all unique message IDs
            all_messages = []
            for messages in search_results.values():
                all_messages.extend(messages)

            # Remove duplicates
            seen_ids = set()
            unique_messages = []
            for msg in all_messages:
                if msg["id"] not in seen_ids:
                    unique_messages.append(msg)
                    seen_ids.add(msg["id"])

            messages = unique_messages[: args.max_results]

        else:
            # Single query operation
            query = build_search_query(args.query, args.days)
            logging.info(f"Searching with query: '{query}'")

            messages = gmail_client.list_messages(query=query, max_results=args.max_results)

        logging.info(f"Found {len(messages)} messages to process")

        # Step 4: Process messages in batches
        if messages:
            detailed_messages = process_messages(
                gmail_client=gmail_client,
                messages=messages,
                format_type=args.format,
                headers=args.headers,
                batch_size=min(args.batch_size, 100),
            )

            # Step 5: Extract useful information
            processed_messages = extract_message_info(detailed_messages, args.headers)

            # Step 6: Generate summary
            generate_summary(processed_messages)

            # Step 7: Save results if requested
            if args.output:
                save_results(processed_messages, args.output)

        logging.info("Gmail API Batch Request Example completed successfully")

    except GmailBatchError as e:
        logging.error(f"Gmail API error: {e}")
        sys.exit(1)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        logging.error("Please ensure the credentials file exists and is valid")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        if args.verbose:
            logging.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
