# OpenSearch Integration for AugmentToolkit

AugmentToolkit now supports reading data directly from OpenSearch indices, allowing you to create LLM fine-tuning datasets from data stored in OpenSearch/Elasticsearch.

## Overview

The OpenSearch integration enables AugmentToolkit to:
- Connect to OpenSearch/Elasticsearch clusters
- Query indices with custom DSL queries
- Extract and process documents for dataset generation
- Support various authentication methods
- Automatically chunk large documents
- Cache results for efficient reprocessing

## Installation

Install the required dependency:

```bash
pip install opensearch-py
```

Or install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Configuration

### Basic Configuration

Add an `opensearch_config` section to your YAML configuration file:

```yaml
opensearch_config:
  host: "localhost"
  port: 9200
  index_name: "my_documents"

  # Authentication
  username: "admin"
  password: "admin"

  # SSL/TLS
  use_ssl: true
  verify_certs: true
```

### Input Directory Configuration

To use OpenSearch as an input source, configure your `input_dirs` section:

```yaml
path:
  input_dirs:
    - use_opensearch_input: true
      opensearch_config_ref: "opensearch_config"
      input_dir_name: "opensearch_documents"
      variation_generation_counts: 4
      # ... other settings
```

### Full Example

See `external_configs/opensearch_example.yaml` for a complete example configuration.

## Configuration Options

### OpenSearch Connection

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `host` | string | `"localhost"` | OpenSearch host address |
| `port` | integer | `9200` | OpenSearch port |
| `index_name` | string | **required** | Name of the index to query |

### Authentication

OpenSearch integration supports multiple authentication methods:

**Basic Authentication:**
```yaml
opensearch_config:
  username: "admin"
  password: "admin"
```

**API Key Authentication:**
```yaml
opensearch_config:
  api_key: "your-api-key-here"
```

### SSL/TLS Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_ssl` | boolean | `true` | Use SSL/TLS for connection |
| `verify_certs` | boolean | `true` | Verify SSL certificates |
| `ca_certs` | string | `null` | Path to CA certificates file |

### Query Configuration

**Match All (Default):**
```yaml
opensearch_config:
  query:
    match_all: {}
```

**Custom Query:**
```yaml
opensearch_config:
  query:
    bool:
      must:
        - match:
            category: "research"
        - range:
            date:
              gte: "2023-01-01"
```

### Field Extraction

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `text_field` | string | `"content"` | Primary field containing text |
| `fields_to_extract` | list | `null` | List of fields to combine (e.g., `["title", "content", "summary"]`) |

**Single Field:**
```yaml
opensearch_config:
  text_field: "content"
  fields_to_extract: null
```

**Multiple Fields:**
```yaml
opensearch_config:
  text_field: "content"
  fields_to_extract: ["title", "content", "summary", "description"]
```

When multiple fields are specified, they are combined with double newlines (`\n\n`).

### Document Limits

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_documents` | integer | `null` | Maximum documents to retrieve (unlimited if null) |

### Chunking

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_opensearch_results` | boolean | `false` | Whether to chunk the results |
| `chunk_size` | integer | `1500` | Token size for chunks if chunking is enabled |

**Enable Chunking:**
```yaml
opensearch_config:
  chunk_opensearch_results: true
  chunk_size: 1500
```

## Usage Examples

### Example 1: Basic Local OpenSearch

```yaml
opensearch_config:
  host: "localhost"
  port: 9200
  index_name: "documents"
  username: "admin"
  password: "admin"
  use_ssl: false
  verify_certs: false
  chunk_opensearch_results: true
  chunk_size: 2000
```

### Example 2: AWS OpenSearch Service

```yaml
opensearch_config:
  host: "search-mydomain-xxx.us-east-1.es.amazonaws.com"
  port: 443
  index_name: "my-index"
  use_ssl: true
  verify_certs: true
  api_key: "your-api-key"
  max_documents: 10000
  fields_to_extract: ["title", "content", "abstract"]
```

### Example 3: Custom Query with Filters

```yaml
opensearch_config:
  host: "opensearch.example.com"
  port: 9200
  index_name: "articles"
  username: "user"
  password: "pass"
  query:
    bool:
      must:
        - match:
            status: "published"
        - range:
            publication_date:
              gte: "2023-01-01"
              lte: "2024-12-31"
      must_not:
        - match:
            category: "draft"
  fields_to_extract: ["title", "body", "summary"]
  chunk_opensearch_results: true
```

### Example 4: Mixing OpenSearch and File-Based Inputs

You can use both OpenSearch and traditional file-based inputs in the same configuration:

```yaml
path:
  input_dirs:
    # OpenSearch input
    - use_opensearch_input: true
      opensearch_config_ref: "opensearch_config"
      input_dir_name: "opensearch_data"
      variation_generation_counts: 4
      # ... other settings

    # File-based input
    - path: ./inputs/local_files
      use_opensearch_input: false
      variation_generation_counts: 4
      # ... other settings
```

## How It Works

1. **Connection**: AugmentToolkit connects to the specified OpenSearch cluster using the provided credentials
2. **Query Execution**: The configured query is executed against the specified index
3. **Data Extraction**: Documents are retrieved and text is extracted from the specified fields
4. **Formatting**: Each document is converted to the standard AugmentToolkit format:
   - `text`: Combined text from extracted fields
   - `metadata`: Index name and document ID (e.g., `"documents:abc123"`)
5. **Chunking** (optional): If enabled, long documents are split into chunks based on token count
6. **Caching**: Results are cached based on query parameters for efficient reprocessing
7. **Processing**: The extracted data flows through the standard AugmentToolkit pipeline

## Metadata

Each OpenSearch document is assigned metadata in the format:
```
{index_name}:{document_id}
```

For example:
- `"articles:doc_12345"`
- `"research_papers:paper_67890"`

This metadata is preserved throughout the pipeline and can be used for source tracking.

## Caching

OpenSearch results are automatically cached to avoid re-querying the same data. The cache key is generated from:
- Host and port
- Index name
- Query DSL
- Fields to extract
- Maximum documents
- Chunking settings

Cached files are stored in the output directory with names like:
```
opensearch_cache_{hash}.json
```

To force a re-fetch, delete the cache file or change any of the configuration parameters.

## Performance Tips

1. **Use field filtering**: Only extract the fields you need using `fields_to_extract`
2. **Set document limits**: Use `max_documents` to limit the initial dataset size during testing
3. **Enable caching**: Use `output_dir` parameter to enable result caching
4. **Optimize queries**: Use filters and queries efficiently to reduce retrieved data
5. **Chunk large documents**: Enable `chunk_opensearch_results` for better processing

## Troubleshooting

### Connection Errors

**Problem**: Cannot connect to OpenSearch
```
Error connecting to or querying OpenSearch: Could not connect to OpenSearch at host:port
```

**Solutions**:
- Verify host and port are correct
- Check firewall/network settings
- Ensure OpenSearch is running
- Verify SSL settings match your cluster configuration

### Authentication Errors

**Problem**: Authentication failed
```
Error: 401 Unauthorized
```

**Solutions**:
- Verify username and password are correct
- Try using API key authentication instead
- Check user permissions in OpenSearch

### SSL/TLS Errors

**Problem**: SSL certificate verification failed
```
Error: SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions**:
- Set `verify_certs: false` for testing (not recommended for production)
- Provide correct CA certificates path using `ca_certs`
- Ensure SSL certificates are valid

### No Documents Retrieved

**Problem**: Query returns 0 documents
```
Retrieved 0 documents from OpenSearch
```

**Solutions**:
- Verify index name is correct
- Check query syntax
- Ensure documents match your query criteria
- Try using `{"match_all": {}}` query to retrieve all documents

### Missing opensearch-py

**Problem**: OpenSearch support not available
```
ImportError: OpenSearch support is not available. Install opensearch-py
```

**Solution**:
```bash
pip install opensearch-py
```

## Python API Usage

You can also use the OpenSearch reader directly in Python:

```python
from generation.core_components.chunking import read_from_opensearch

# Read from OpenSearch
data = read_from_opensearch(
    host="localhost",
    port=9200,
    index_name="documents",
    query={"match": {"category": "research"}},
    fields_to_extract=["title", "content"],
    max_documents=1000,
    username="admin",
    password="admin",
    chunk_opensearch_results=True,
    chunk_size=1500,
    output_dir="./cache"
)

# data is a list of dicts: [{"text": "...", "metadata": "..."}, ...]
```

## Security Considerations

1. **Never commit credentials**: Use environment variables or secure vaults for credentials
2. **Use SSL/TLS**: Always enable `use_ssl: true` for production
3. **Verify certificates**: Keep `verify_certs: true` for production
4. **Limit access**: Use read-only credentials with minimal required permissions
5. **Network security**: Use VPNs or private networks when possible

## Compatibility

- **OpenSearch**: 1.x, 2.x
- **Elasticsearch**: 7.x, 8.x (using opensearch-py with compatibility mode)
- **AWS OpenSearch Service**: Fully supported
- **Elastic Cloud**: Supported

## Support

For issues, questions, or contributions related to OpenSearch integration:
1. Check this documentation
2. Review the example configuration: `external_configs/opensearch_example.yaml`
3. Check the implementation: `generation/core_components/chunking.py`
4. Open an issue on GitHub

## License

OpenSearch integration is part of AugmentToolkit and follows the same license.
