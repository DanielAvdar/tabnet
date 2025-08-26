Gmail API Batch Requests Guide
==============================

This guide demonstrates how to use Gmail API batch requests in Python for efficient email fetching.

Overview
--------

The Gmail API provides batch capabilities that allow you to send multiple requests in a single HTTP call,
significantly improving performance when dealing with large volumes of emails. This is particularly useful
when you need to fetch details for many messages or perform multiple search queries.

Features
--------

* **Batch Message Fetching**: Retrieve details for multiple messages in a single API call
* **Efficient Message Listing**: List messages with pagination support
* **Multiple Search Queries**: Perform batch searches across different criteria
* **Error Handling & Retry Logic**: Robust error handling with exponential backoff
* **Rate Limit Management**: Automatic handling of Gmail API rate limits
* **Flexible Authentication**: Support for OAuth2 authentication flow

Installation
------------

First, install the required Google API client libraries:

.. code-block:: bash

    pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2

Setup Authentication
--------------------

Before using the Gmail API, you need to set up authentication:

1. Go to the `Google Cloud Console <https://console.cloud.google.com/>`_
2. Create a new project or select an existing one
3. Enable the Gmail API
4. Create credentials (OAuth 2.0 Client IDs)
5. Download the credentials JSON file

Basic Usage
-----------

Here's a complete example of how to use the Gmail API batch utilities:

.. code-block:: python

    from pytorch_tabnet.utils import GmailAPIClient, create_credentials

    # Step 1: Create credentials
    credentials = create_credentials('path/to/credentials.json')

    # Step 2: Initialize Gmail API client
    gmail_client = GmailAPIClient(credentials)

    # Step 3: List messages
    messages = gmail_client.list_messages(
        query='from:important@example.com',
        max_results=100
    )
    print(f"Found {len(messages)} messages")

    # Step 4: Fetch message details in batches
    if messages:
        message_ids = [msg['id'] for msg in messages]
        detailed_messages = gmail_client.fetch_messages_batch(
            message_ids=message_ids,
            format_type='metadata',
            metadata_headers=['From', 'Subject', 'Date'],
            batch_size=25
        )

        print(f"Fetched details for {len(detailed_messages)} messages")

Advanced Usage
--------------

Batch Search Operations
~~~~~~~~~~~~~~~~~~~~~~~

Perform multiple searches simultaneously:

.. code-block:: python

    search_results = gmail_client.search_messages_batch([
        'from:notifications@github.com',
        'from:noreply@stackoverflow.com',
        'has:attachment filename:pdf'
    ])

    for query, results in search_results.items():
        print(f"Query '{query}' returned {len(results)} messages")

Custom Retry Logic
~~~~~~~~~~~~~~~~~~

Use the retry utility for robust batch operations:

.. code-block:: python

    from pytorch_tabnet.utils import batch_operation_with_retry

    def process_message_batch(message_ids):
        return gmail_client.fetch_messages_batch(message_ids)

    all_message_ids = [...]  # List of all message IDs
    results = batch_operation_with_retry(
        process_message_batch,
        all_message_ids,
        batch_size=50,
        max_retries=3,
        retry_delay=2.0
    )

Extracting Message Headers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extract specific headers from messages:

.. code-block:: python

    from pytorch_tabnet.utils import get_message_headers

    for message in detailed_messages:
        headers = get_message_headers(message, ['From', 'Subject', 'Date', 'To'])
        print(f"From: {headers.get('From', 'Unknown')}")
        print(f"Subject: {headers.get('Subject', 'No Subject')}")
        print(f"Date: {headers.get('Date', 'Unknown')}")
        print("---")

API Reference
-------------

GmailAPIClient
~~~~~~~~~~~~~~

.. autoclass:: pytorch_tabnet.utils.gmail_api.GmailAPIClient
    :members:
    :undoc-members:
    :show-inheritance:

Utility Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: pytorch_tabnet.utils.gmail_api.create_credentials

.. autofunction:: pytorch_tabnet.utils.gmail_api.get_message_headers

.. autofunction:: pytorch_tabnet.utils.gmail_api.batch_operation_with_retry

Best Practices
--------------

1. **Use Appropriate Batch Sizes**:

   * Gmail API allows up to 100 requests per batch
   * Recommended batch size: 25-50 for message fetching
   * Adjust based on your use case and rate limits

2. **Handle Rate Limits**:

   * The Gmail API has usage quotas and rate limits
   * Use exponential backoff for retries
   * Add delays between large batch operations

3. **Choose Appropriate Message Formats**:

   * ``minimal``: Only message ID and labels
   * ``full``: Complete message data (use sparingly)
   * ``raw``: Raw MIME message
   * ``metadata``: Message metadata and headers (recommended)

4. **Error Handling**:

   * Always wrap API calls in try-catch blocks
   * Log failed operations for debugging
   * Implement retry logic for transient failures

5. **Authentication**:

   * Store credentials securely
   * Refresh tokens when they expire
   * Use appropriate OAuth scopes

Example: Complete Email Processing Pipeline
-------------------------------------------

Here's a comprehensive example that demonstrates a complete email processing pipeline:

.. code-block:: python

    import json
    import logging
    from datetime import datetime, timedelta
    from pytorch_tabnet.utils import (
        GmailAPIClient,
        create_credentials,
        get_message_headers,
        batch_operation_with_retry
    )

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    def process_emails_pipeline():
        \"\"\"Complete email processing pipeline example.\"\"\"
        try:
            # Step 1: Authentication
            credentials = create_credentials(
                'credentials.json',
                token_file='gmail_token.json'
            )

            gmail_client = GmailAPIClient(credentials)

            # Step 2: Define search criteria
            # Last 7 days
            date_7_days_ago = (datetime.now() - timedelta(days=7)).strftime('%Y/%m/%d')

            search_queries = [
                f'after:{date_7_days_ago} from:github.com',
                f'after:{date_7_days_ago} has:attachment',
                f'after:{date_7_days_ago} is:important'
            ]

            # Step 3: Batch search operations
            logger.info("Starting batch search operations...")
            search_results = gmail_client.search_messages_batch(
                search_queries,
                max_results_per_query=200
            )

            # Step 4: Collect all unique message IDs
            all_message_ids = set()
            for query, messages in search_results.items():
                message_ids = {msg['id'] for msg in messages}
                all_message_ids.update(message_ids)
                logger.info(f"Query '{query}': {len(messages)} messages")

            logger.info(f"Total unique messages: {len(all_message_ids)}")

            # Step 5: Batch fetch message details
            if all_message_ids:
                def fetch_batch(message_ids):
                    return gmail_client.fetch_messages_batch(
                        message_ids,
                        format_type='metadata',
                        metadata_headers=['From', 'Subject', 'Date', 'To', 'Cc'],
                        batch_size=30
                    )

                detailed_messages = batch_operation_with_retry(
                    fetch_batch,
                    list(all_message_ids),
                    batch_size=120,  # Process 120 IDs at a time (4 API batches)
                    max_retries=3,
                    retry_delay=1.0
                )

                logger.info(f"Fetched details for {len(detailed_messages)} messages")

                # Step 6: Process and analyze messages
                processed_emails = []
                for message in detailed_messages:
                    headers = get_message_headers(
                        message,
                        ['From', 'Subject', 'Date', 'To', 'Cc']
                    )

                    # Extract useful information
                    email_data = {
                        'id': message.get('id'),
                        'thread_id': message.get('threadId'),
                        'from': headers.get('From'),
                        'subject': headers.get('Subject'),
                        'date': headers.get('Date'),
                        'to': headers.get('To'),
                        'cc': headers.get('Cc'),
                        'labels': message.get('labelIds', [])
                    }
                    processed_emails.append(email_data)

                # Step 7: Save results
                with open('processed_emails.json', 'w') as f:
                    json.dump(processed_emails, f, indent=2, default=str)

                logger.info(f"Processed {len(processed_emails)} emails")

                # Step 8: Generate summary statistics
                generate_email_statistics(processed_emails)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    def generate_email_statistics(emails):
        \"\"\"Generate and display email statistics.\"\"\"
        from collections import Counter

        # Count emails by sender domain
        domains = []
        for email in emails:
            if email['from']:
                domain = email['from'].split('@')[-1].split('>')[-1]
                domains.append(domain)

        domain_counts = Counter(domains)

        print("\\n=== Email Statistics ===")
        print(f"Total emails processed: {len(emails)}")
        print(f"\\nTop 10 sender domains:")
        for domain, count in domain_counts.most_common(10):
            print(f"  {domain}: {count}")

        # Count by labels
        all_labels = []
        for email in emails:
            all_labels.extend(email.get('labels', []))

        label_counts = Counter(all_labels)
        print(f"\\nTop 10 labels:")
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count}")

    if __name__ == "__main__":
        process_emails_pipeline()

Performance Considerations
--------------------------

**Batch Size Optimization**:

* Gmail API allows max 100 requests per batch
* Optimal batch size depends on:

  * Message size and complexity
  * Network latency
  * Available memory
  * API quota limits

**Rate Limit Management**:

* Gmail API quota: 250 quota units per user per second
* Different operations consume different quota units:

  * messages.list: 5 units
  * messages.get: 5 units
  * Batch requests: Sum of individual request costs

**Memory Considerations**:

* Large batches can consume significant memory
* Consider processing results incrementally
* Use generators for very large datasets

Common Error Scenarios
----------------------

**Authentication Errors**:

.. code-block:: python

    try:
        credentials = create_credentials('credentials.json')
    except Exception as e:
        print(f"Authentication failed: {e}")
        # Handle re-authentication

**Rate Limit Exceeded**:

.. code-block:: python

    from googleapiclient.errors import HttpError

    try:
        messages = gmail_client.list_messages(max_results=1000)
    except HttpError as e:
        if e.resp.status == 429:  # Rate limit exceeded
            print("Rate limit exceeded, waiting...")
            time.sleep(60)  # Wait and retry
        else:
            print(f"HTTP error: {e}")

**Network Timeouts**:

.. code-block:: python

    import socket

    try:
        detailed_messages = gmail_client.fetch_messages_batch(message_ids)
    except socket.timeout:
        print("Network timeout, retrying with smaller batch size...")
        # Retry with smaller batch size

Limitations and Caveats
-----------------------

1. **API Quotas**: Gmail API has daily and per-second quotas that may limit throughput
2. **Message Size**: Very large messages may cause memory issues in batch operations
3. **Authentication**: OAuth tokens expire and need refreshing
4. **Scope Limitations**: Read-only access requires ``gmail.readonly`` scope
5. **Batch Limits**: Maximum 100 requests per batch, maximum 1000 batches per request

Security Considerations
-----------------------

* Store credentials securely (never commit to version control)
* Use minimal required OAuth scopes
* Implement proper token refresh mechanisms
* Log access for audit purposes
* Consider using service accounts for server applications

Troubleshooting
---------------

**Common Issues**:

1. **"Credentials not found"**: Ensure credentials.json exists and is valid
2. **"Access denied"**: Check OAuth scopes and user permissions
3. **"Quota exceeded"**: Implement rate limiting and retry logic
4. **"Invalid message ID"**: Verify message IDs are current and accessible
5. **"Batch request failed"**: Check individual request errors in batch response

**Debug Mode**:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # This will show detailed API request/response information

For additional support, refer to the official `Gmail API documentation <https://developers.google.com/workspace/gmail/api>`_.
