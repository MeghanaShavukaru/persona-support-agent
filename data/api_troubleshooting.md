# API Troubleshooting Guide

## Authentication Errors

When a request receives a `401 Unauthorized`, verify the API key, bearer token expiration, and header format.

Example header:

```
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

If the token is valid, confirm that the requested endpoint is enabled for the account and that the client IP is allowed.

## Rate Limits

The API enforces rate limits per account. If you receive a `429 Too Many Requests`, retry after the `Retry-After` header.

## Common Header Parameters

- `Authorization`: bearer token
- `Content-Type`: application/json
- `X-Request-ID`: optional tracing header

For API integrations, always send JSON payloads and verify that hostnames match the registered application.
