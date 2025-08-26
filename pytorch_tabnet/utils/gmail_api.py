"""Gmail API batch utilities for efficient email fetching.

This module provides utilities for sending batch requests to the Gmail API
using Python for faster and more reliable email fetching. It includes
functions for listing messages, fetching message details, and handling
batch operations with proper error handling and retry logic.

References:
- Gmail API Batch Requests: https://developers.google.com/workspace/gmail/api/guides/batch
- Gmail API List Messages: https://developers.google.com/workspace/gmail/api/guides/list-messages

Required libraries:
- google-api-python-client
- google-auth
- google-auth-oauthlib (for OAuth2 authentication)
- google-auth-httplib2
"""

import logging
import time
from typing import Any, Dict, List, Optional

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient import discovery
    from googleapiclient.errors import HttpError
    from googleapiclient.http import BatchHttpRequest

    GMAIL_API_AVAILABLE = True
except ImportError:
    GMAIL_API_AVAILABLE = False
    # Define placeholder classes for type hints when dependencies are not available
    BatchHttpRequest = None
    discovery = None
    HttpError = Exception


logger = logging.getLogger(__name__)


class GmailBatchError(Exception):
    """Custom exception for Gmail batch request errors."""

    pass


class GmailAPIClient:
    """Gmail API client with batch request capabilities.

    This class provides methods to interact with the Gmail API using batch requests
    for improved performance when dealing with multiple operations.

    Args:
        credentials: Google OAuth2 credentials object
        user_id: Gmail user ID (default: 'me' for authenticated user)

    Example:
        >>> from pytorch_tabnet.utils.gmail_api import GmailAPIClient
        >>> client = GmailAPIClient(credentials)
        >>> messages = client.list_messages(max_results=100)
        >>> batch_details = client.fetch_messages_batch([msg['id'] for msg in messages])
    """

    def __init__(self, credentials: Any, user_id: str = "me"):
        """Initialize Gmail API client."""
        if not GMAIL_API_AVAILABLE:
            raise ImportError(
                "Gmail API dependencies not available. Please install: "
                "pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2"
            )

        self.credentials = credentials
        self.user_id = user_id
        self.service = discovery.build("gmail", "v1", credentials=credentials)

    def list_messages(
        self, query: str = "", max_results: int = 100, label_ids: Optional[List[str]] = None, include_spam_trash: bool = False
    ) -> List[Dict[str, Any]]:
        """List messages from Gmail using the messages.list API.

        Args:
            query: Gmail search query (e.g., 'from:example@gmail.com')
            max_results: Maximum number of messages to retrieve
            label_ids: List of label IDs to filter messages
            include_spam_trash: Whether to include spam and trash messages

        Returns:
            List of message dictionaries with 'id' and 'threadId' fields

        Raises:
            GmailBatchError: If the API request fails
        """
        try:
            messages = []
            request = (
                self.service.users()
                .messages()
                .list(
                    userId=self.user_id,
                    q=query,
                    maxResults=min(max_results, 500),  # Gmail API limit is 500 per request
                    labelIds=label_ids,
                    includeSpamTrash=include_spam_trash,
                )
            )

            while request and len(messages) < max_results:
                response = request.execute()
                if "messages" in response:
                    messages.extend(response["messages"][: max_results - len(messages)])

                if len(messages) >= max_results:
                    break

                request = self.service.users().messages().list_next(request, response)

            logger.info(f"Listed {len(messages)} messages")
            return messages

        except HttpError as error:
            raise GmailBatchError(f"Failed to list messages: {error}") from error

    def fetch_messages_batch(
        self, message_ids: List[str], format_type: str = "full", metadata_headers: Optional[List[str]] = None, batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Fetch multiple messages using batch requests for improved performance.

        Args:
            message_ids: List of Gmail message IDs to fetch
            format_type: Message format ('minimal', 'full', 'raw', 'metadata')
            metadata_headers: List of headers to include when format is 'metadata'
            batch_size: Number of requests per batch (max 100)

        Returns:
            List of message details dictionaries

        Raises:
            GmailBatchError: If batch requests fail
        """
        if batch_size > 100:
            batch_size = 100
            logger.warning("Batch size limited to 100 (Gmail API limit)")

        all_messages = []

        # Process messages in batches
        for i in range(0, len(message_ids), batch_size):
            batch_ids = message_ids[i : i + batch_size]
            batch_messages = self._fetch_batch_chunk(batch_ids, format_type, metadata_headers)
            all_messages.extend(batch_messages)

            # Add small delay between batches to respect rate limits
            if i + batch_size < len(message_ids):
                time.sleep(0.1)

        logger.info(f"Fetched {len(all_messages)} messages in batches")
        return all_messages

    def _fetch_batch_chunk(self, message_ids: List[str], format_type: str, metadata_headers: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Fetch a single batch chunk of messages.

        Args:
            message_ids: List of message IDs for this batch
            format_type: Message format type
            metadata_headers: Headers to include for metadata format

        Returns:
            List of message dictionaries
        """
        batch_request = BatchHttpRequest()
        batch_results = []
        batch_errors = []

        def callback(request_id: str, response: Dict[str, Any], exception: Optional[Exception]):
            """Callback function for batch request responses."""
            if exception:
                batch_errors.append({"message_id": request_id, "error": str(exception)})
                logger.warning(f"Error fetching message {request_id}: {exception}")
            else:
                batch_results.append(response)

        # Add requests to batch
        for message_id in message_ids:
            request = (
                self.service.users()
                .messages()
                .get(userId=self.user_id, id=message_id, format=format_type, metadataHeaders=metadata_headers)
            )
            batch_request.add(request, callback=callback, request_id=message_id)

        # Execute batch request
        try:
            batch_request.execute()
        except HttpError as error:
            raise GmailBatchError(f"Batch request failed: {error}") from error

        if batch_errors:
            logger.warning(f"Encountered {len(batch_errors)} errors in batch")

        return batch_results

    def search_messages_batch(self, queries: List[str], max_results_per_query: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """Perform multiple message searches using batch requests.

        Args:
            queries: List of Gmail search queries
            max_results_per_query: Maximum results per query

        Returns:
            Dictionary mapping queries to their message results
        """
        results = {}

        for query in queries:
            try:
                messages = self.list_messages(query=query, max_results=max_results_per_query)
                results[query] = messages
            except GmailBatchError as e:
                logger.error(f"Failed to search for query '{query}': {e}")
                results[query] = []

        return results


def create_credentials(credentials_file: str, token_file: str = "token.json", scopes: Optional[List[str]] = None) -> Any:
    """Create Gmail API credentials using OAuth2 flow.

    Args:
        credentials_file: Path to OAuth2 credentials JSON file
        token_file: Path to store/load the access token
        scopes: List of OAuth2 scopes (default: readonly Gmail access)

    Returns:
        Google OAuth2 credentials object

    Example:
        >>> credentials = create_credentials('credentials.json')
        >>> client = GmailAPIClient(credentials)
    """
    if not GMAIL_API_AVAILABLE:
        raise ImportError(
            "Gmail API dependencies not available. Please install: "
            "pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2"
        )

    if scopes is None:
        scopes = ["https://www.googleapis.com/auth/gmail.readonly"]

    creds = None

    # Load existing token if available
    try:
        if token_file:
            creds = Credentials.from_authorized_user_file(token_file, scopes)
    except Exception:
        pass

    # If no valid credentials, start OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, scopes)
            creds = flow.run_local_server(port=0)

        # Save credentials for future use
        if token_file:
            with open(token_file, "w") as token:
                token.write(creds.to_json())

    return creds


def get_message_headers(message: Dict[str, Any], header_names: List[str]) -> Dict[str, str]:
    """Extract specific headers from a Gmail message.

    Args:
        message: Gmail message dictionary
        header_names: List of header names to extract (e.g., ['From', 'Subject'])

    Returns:
        Dictionary mapping header names to their values
    """
    headers = {}
    if "payload" in message and "headers" in message["payload"]:
        for header in message["payload"]["headers"]:
            if header["name"] in header_names:
                headers[header["name"]] = header["value"]
    return headers


def batch_operation_with_retry(
    operation_func, items: List[Any], batch_size: int = 100, max_retries: int = 3, retry_delay: float = 1.0
) -> List[Any]:
    """Execute batch operations with retry logic.

    Args:
        operation_func: Function to execute for each batch
        items: List of items to process
        batch_size: Size of each batch
        max_retries: Maximum number of retries for failed batches
        retry_delay: Delay between retries in seconds

    Returns:
        List of results from all successful operations
    """
    results = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        for attempt in range(max_retries):
            try:
                batch_results = operation_func(batch)
                results.extend(batch_results)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Batch operation failed after {max_retries} attempts: {e}")
                    raise
                else:
                    logger.warning(f"Batch operation attempt {attempt + 1} failed, retrying: {e}")
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff

    return results


# Example usage and best practices
def example_usage():
    """Example demonstrating Gmail API batch operations.

    This function shows how to use the Gmail API batch utilities
    to efficiently fetch emails with proper error handling.
    """
    if not GMAIL_API_AVAILABLE:
        print("Gmail API dependencies not installed. Please install required packages:")
        print("pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2")
        return

    try:
        # Step 1: Create credentials
        credentials = create_credentials("path/to/credentials.json")

        # Step 2: Initialize Gmail API client
        gmail_client = GmailAPIClient(credentials)

        # Step 3: List messages with a query
        messages = gmail_client.list_messages(query="from:important@example.com", max_results=50)
        print(f"Found {len(messages)} messages")

        # Step 4: Fetch message details in batches
        if messages:
            message_ids = [msg["id"] for msg in messages]
            detailed_messages = gmail_client.fetch_messages_batch(
                message_ids=message_ids, format_type="metadata", metadata_headers=["From", "Subject", "Date"], batch_size=25
            )

            print(f"Fetched details for {len(detailed_messages)} messages")

            # Extract and display headers
            for message in detailed_messages[:5]:  # Show first 5
                headers = get_message_headers(message, ["From", "Subject", "Date"])
                print(f"From: {headers.get('From', 'Unknown')}")
                print(f"Subject: {headers.get('Subject', 'No Subject')}")
                print(f"Date: {headers.get('Date', 'Unknown')}")
                print("---")

        # Step 5: Perform multiple searches
        search_results = gmail_client.search_messages_batch([
            "from:notifications@github.com",
            "from:noreply@stackoverflow.com",
            "has:attachment",
        ])

        for query, results in search_results.items():
            print(f"Query '{query}' returned {len(results)} messages")

    except Exception as e:
        print(f"Error in example usage: {e}")


if __name__ == "__main__":
    example_usage()
