"""Tests for Gmail API batch utilities."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from pytorch_tabnet.utils.gmail_api import (
    GMAIL_API_AVAILABLE,
    GmailAPIClient,
    GmailBatchError,
    batch_operation_with_retry,
    create_credentials,
    get_message_headers,
)


class TestGmailAPIClient(unittest.TestCase):
    """Test cases for GmailAPIClient class."""

    def setUp(self):
        """Set up test fixtures."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        # Mock credentials and service
        self.mock_credentials = MagicMock()
        self.mock_service = MagicMock()

        with patch("pytorch_tabnet.utils.gmail_api.discovery.build") as mock_build:
            mock_build.return_value = self.mock_service
            self.client = GmailAPIClient(self.mock_credentials)

    def test_init_without_dependencies(self):
        """Test initialization when Gmail API dependencies are not available."""
        with patch("pytorch_tabnet.utils.gmail_api.GMAIL_API_AVAILABLE", False):
            with self.assertRaises(ImportError) as context:
                GmailAPIClient(Mock())
            self.assertIn("Gmail API dependencies not available", str(context.exception))

    def test_init_with_dependencies(self):
        """Test successful initialization with dependencies."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        with patch("pytorch_tabnet.utils.gmail_api.discovery.build") as mock_build:
            mock_service = MagicMock()
            mock_build.return_value = mock_service

            credentials = MagicMock()
            client = GmailAPIClient(credentials, user_id="test@example.com")

            self.assertEqual(client.credentials, credentials)
            self.assertEqual(client.user_id, "test@example.com")
            self.assertEqual(client.service, mock_service)

    def test_list_messages_success(self):
        """Test successful message listing."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        # Mock API response
        mock_messages = [{"id": "msg1", "threadId": "thread1"}, {"id": "msg2", "threadId": "thread2"}]
        mock_response = {"messages": mock_messages}

        mock_request = MagicMock()
        mock_request.execute.return_value = mock_response

        self.mock_service.users().messages().list.return_value = mock_request
        self.mock_service.users().messages().list_next.return_value = None

        # Test the method
        result = self.client.list_messages(query="test query", max_results=10)

        # Assertions
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "msg1")
        self.assertEqual(result[1]["id"], "msg2")

        # Verify API call parameters
        self.mock_service.users().messages().list.assert_called_once_with(
            userId="me", q="test query", maxResults=10, labelIds=None, includeSpamTrash=False
        )

    def test_list_messages_with_pagination(self):
        """Test message listing with pagination."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        # Mock first page
        mock_messages_page1 = [{"id": f"msg{i}", "threadId": f"thread{i}"} for i in range(1, 4)]
        mock_response_page1 = {"messages": mock_messages_page1}

        # Mock second page
        mock_messages_page2 = [{"id": f"msg{i}", "threadId": f"thread{i}"} for i in range(4, 6)]
        mock_response_page2 = {"messages": mock_messages_page2}

        mock_request1 = MagicMock()
        mock_request1.execute.return_value = mock_response_page1

        mock_request2 = MagicMock()
        mock_request2.execute.return_value = mock_response_page2

        self.mock_service.users().messages().list.return_value = mock_request1
        self.mock_service.users().messages().list_next.side_effect = [mock_request2, None]

        # Test the method
        result = self.client.list_messages(max_results=10)

        # Assertions
        self.assertEqual(len(result), 5)
        self.assertEqual(result[0]["id"], "msg1")
        self.assertEqual(result[4]["id"], "msg5")

    def test_list_messages_http_error(self):
        """Test message listing with HTTP error."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        from googleapiclient.errors import HttpError

        mock_request = MagicMock()
        mock_request.execute.side_effect = HttpError(Mock(status=400), b"Bad Request")

        self.mock_service.users().messages().list.return_value = mock_request

        with self.assertRaises(GmailBatchError) as context:
            self.client.list_messages()

        self.assertIn("Failed to list messages", str(context.exception))

    def test_fetch_messages_batch_success(self):
        """Test successful batch message fetching."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        message_ids = ["msg1", "msg2", "msg3"]
        expected_messages = [
            {"id": "msg1", "payload": {"headers": []}},
            {"id": "msg2", "payload": {"headers": []}},
            {"id": "msg3", "payload": {"headers": []}},
        ]

        # Mock the _fetch_batch_chunk method
        with patch.object(self.client, "_fetch_batch_chunk") as mock_fetch:
            mock_fetch.return_value = expected_messages

            result = self.client.fetch_messages_batch(message_ids)

            self.assertEqual(len(result), 3)
            self.assertEqual(result, expected_messages)
            mock_fetch.assert_called_once_with(message_ids, "full", None)

    def test_fetch_messages_batch_large_batch_size(self):
        """Test batch fetching with batch size larger than limit."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        message_ids = ["msg1", "msg2"]

        with patch.object(self.client, "_fetch_batch_chunk") as mock_fetch:
            mock_fetch.return_value = []

            self.client.fetch_messages_batch(message_ids, batch_size=150)

            # Should be called with batch size limited to 100
            mock_fetch.assert_called_once_with(message_ids, "full", None)

    def test_search_messages_batch(self):
        """Test batch message searching."""
        if not GMAIL_API_AVAILABLE:
            self.skipTest("Gmail API dependencies not available")

        queries = ["from:test1@example.com", "from:test2@example.com"]
        expected_results = {"from:test1@example.com": [{"id": "msg1"}], "from:test2@example.com": [{"id": "msg2"}]}

        with patch.object(self.client, "list_messages") as mock_list:
            mock_list.side_effect = lambda query, max_results: expected_results[query]

            result = self.client.search_messages_batch(queries, max_results_per_query=50)

            self.assertEqual(result, expected_results)
            self.assertEqual(mock_list.call_count, 2)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""

    def test_get_message_headers(self):
        """Test header extraction from Gmail message."""
        message = {
            "payload": {
                "headers": [
                    {"name": "From", "value": "test@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    {"name": "Other", "value": "Other Value"},
                ]
            }
        }

        headers = get_message_headers(message, ["From", "Subject", "Missing"])

        self.assertEqual(headers["From"], "test@example.com")
        self.assertEqual(headers["Subject"], "Test Subject")
        self.assertNotIn("Date", headers)
        self.assertNotIn("Missing", headers)

    def test_get_message_headers_no_payload(self):
        """Test header extraction when message has no payload."""
        message = {"id": "test_id"}

        headers = get_message_headers(message, ["From", "Subject"])

        self.assertEqual(headers, {})

    def test_get_message_headers_no_headers(self):
        """Test header extraction when payload has no headers."""
        message = {"payload": {"body": "test body"}}

        headers = get_message_headers(message, ["From", "Subject"])

        self.assertEqual(headers, {})

    def test_batch_operation_with_retry_success(self):
        """Test successful batch operation with retry logic."""
        items = [1, 2, 3, 4, 5]

        def mock_operation(batch):
            return [x * 2 for x in batch]

        result = batch_operation_with_retry(mock_operation, items, batch_size=2)

        self.assertEqual(result, [2, 4, 6, 8, 10])

    def test_batch_operation_with_retry_failure_then_success(self):
        """Test batch operation that fails then succeeds on retry."""
        items = [1, 2, 3]
        call_count = 0

        def mock_operation(batch):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary failure")
            return [x * 2 for x in batch]

        result = batch_operation_with_retry(mock_operation, items, batch_size=3, max_retries=2)

        self.assertEqual(result, [2, 4, 6])
        self.assertEqual(call_count, 2)

    def test_batch_operation_with_retry_permanent_failure(self):
        """Test batch operation that permanently fails."""
        items = [1, 2, 3]

        def mock_operation(batch):
            raise Exception("Permanent failure")

        with self.assertRaises(Exception) as context:
            batch_operation_with_retry(mock_operation, items, max_retries=2)

        self.assertIn("Permanent failure", str(context.exception))

    @unittest.skipUnless(GMAIL_API_AVAILABLE, "Gmail API dependencies not available")
    def test_create_credentials_without_existing_token(self):
        """Test credential creation without existing token."""
        with patch("pytorch_tabnet.utils.gmail_api.Credentials.from_authorized_user_file") as mock_from_file:
            mock_from_file.side_effect = Exception("No file")

            with patch("pytorch_tabnet.utils.gmail_api.InstalledAppFlow.from_client_secrets_file") as mock_flow:
                mock_creds = MagicMock()
                mock_creds.to_json.return_value = '{"token": "test"}'
                mock_flow_instance = MagicMock()
                mock_flow_instance.run_local_server.return_value = mock_creds
                mock_flow.return_value = mock_flow_instance

                with patch("builtins.open", unittest.mock.mock_open()) as mock_open:
                    result = create_credentials("credentials.json", "token.json")

                    self.assertEqual(result, mock_creds)
                    mock_flow.assert_called_once()
                    mock_open.assert_called_once_with("token.json", "w")

    def test_create_credentials_without_dependencies(self):
        """Test credential creation when dependencies are not available."""
        with patch("pytorch_tabnet.utils.gmail_api.GMAIL_API_AVAILABLE", False):
            with self.assertRaises(ImportError) as context:
                create_credentials("credentials.json")
            self.assertIn("Gmail API dependencies not available", str(context.exception))


class TestGmailBatchError(unittest.TestCase):
    """Test cases for GmailBatchError exception."""

    def test_gmail_batch_error(self):
        """Test GmailBatchError exception."""
        error_message = "Test error message"
        error = GmailBatchError(error_message)

        self.assertEqual(str(error), error_message)
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main()
