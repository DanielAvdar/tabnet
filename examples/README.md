# Examples

This directory contains example scripts and notebooks demonstrating various features of pytorch-tabnet.

## Gmail API Batch Requests

### gmail_batch_example.py

A comprehensive example demonstrating how to use the Gmail API batch utilities for efficient email fetching.

**Requirements:**
```bash
pip install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
```

**Setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a project and enable Gmail API
3. Create OAuth 2.0 credentials
4. Download the credentials JSON file

**Usage Examples:**

```bash
# Basic search
python gmail_batch_example.py --credentials credentials.json --query "from:notifications@github.com"

# Search recent emails with attachments
python gmail_batch_example.py --credentials credentials.json --query "has:attachment" --days 7

# Batch search demonstration
python gmail_batch_example.py --credentials credentials.json --batch-search

# Save results to file
python gmail_batch_example.py --credentials credentials.json --query "is:important" --output results.json

# Verbose logging
python gmail_batch_example.py --credentials credentials.json --query "from:example.com" --verbose
```

**Features:**
- OAuth2 authentication flow
- Batch message fetching with retry logic
- Multiple search query processing
- Configurable message formats and headers
- Result export to JSON
- Comprehensive logging and error handling

For detailed documentation, see the [Gmail API Batch Guide](../docs/source/guides/gmail_api_batch.rst).
