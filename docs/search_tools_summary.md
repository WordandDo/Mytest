# Search Tools Summary

## Tool Set
- `文搜文(query, k=1..20, region='cn')` → `TextSearchService.search_with_summaries`
- `web_visit(url, region='cn')` → `TextSearchService.visit_url`
- `image_search_by_text(query, k=1..20)` → `ImageSearchService.search_by_query`
- `reverse_image_search(image_token, k=1..10, messages=messages)` → `ImageSearchService.search_by_image` using token-resolved URL
- `crop_images_by_token(crop_config, messages=messages)` → `ImageProcessor.batch_crop_images` (returns mapping + images list for injection)
- Hidden `upload_file_to_cloud(file_path)` → `CloudStorageService.upload_single_image`

## Input Expectations
- All tools receive simple typed arguments (strings, ints, dict) annotated with `Annotated`/`Field`.
- `web_visit` expects a URL returned by another tool (e.g., `文搜文` or `image_search_by_text`) so it has something to describe.
- `reverse_image_search` / `crop_images_by_token` rely on `ImageProcessor` extracting `<image_x>`/`<obs_x>` tokens from the conversation history passed via `messages`.
- `crop_images_by_token` expects `{'<token>': [left, top, right, bottom]}` crop boxes.

- ## Data Structures / Returns
- Every tool returns a **JSON string** (indent=2); success payloads describe search results, image metadata, crop outputs, or URLs.
- `文搜文` results now include `snippet`, `source`, `image_url`/`imageUrl`, `thumbnail_url`/`thumbnailUrl`, `link` (alias of `url`), `position`, and optional `date`.
- `web_visit` returns a single `{ "text": "<summary>" }` payload with the condensed title/snippet/source text for that URL.
- `image_search_by_text` / `reverse_image_search` results include `title`, `link`, `thumbnail`/`thumbnailUrl`, `image_url`/`imageUrl`, and optional `source`/`domain`/`position`.
- Errors use the common shape `{"status":"error","tool":"<name>","message":"..."}`.
- `crop_images_by_token` returns `{"results": {token: path/url/error, ...}, "images": [...]}`; images are URL (cloud) or base64 (local) depending on `SEARCH_STORAGE_MODE`, and can be injected as `<obs_i>`.
- Hidden `upload_file_to_cloud` returns upload metadata JSON (typically containing a cloud `url`).

## Image Token Handling
- `HttpMCPSearchEnv` injects `<image_k>` markers for task input images and wraps tool-returned images in `<obs_i>...</obs_i>`, which the gateway-side `ImageProcessor` indexes when `reverse_image_search` or `crop_images_by_token` run.
- Tokens are recognized by seeing a `<token>` text entry immediately followed by an `image_url`, allowing `ImageProcessor` to map them without loading the actual image until needed.

## Flow Notes
- HttpMCPEnv `_call_tool_sync` wraps tool output into `{text: "<json string>", images: []}`.
- `HttpMCPSearchEnv` then translates any tool-generated screenshots into tokenized blocks for downstream cropping/image search.
- Running a rollout against the new server (port 9090) requires pointing `HttpMCPEEnv` subclasses at `http://localhost:9090` via `mcp_server_url` and ensuring `gateway_config_searchtools.json` is used so the search_tools module is registered.

## Module Overview
- `src/utils/search_v2/text_search.py` implements `TextSearchService`, which queries SerpAPI for organic results, shapes each item into title/url/snippet/source/image_url/thumbnail_url entries, and exposes helpers for optional content fetches or LLM summaries if necessary.
- `src/utils/search_v2/image_search.py` implements `ImageSearchService`, which normalizes URL/Base64/local inputs (auto-uploading non-HTTP images through `CloudStorageService`) and runs multiple SerpAPI engines (`google_lens`, `google_reverse_image`, `google_images`) to collect titles, thumbnails, image URLs, and links for both reverse and text-based image search flows.
