import base64
import csv
import io
import json
import os
import re
import shutil
import subprocess
import traceback
from io import StringIO
from typing import Any, List, Mapping, NotRequired, TypedDict, cast

from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from PIL import Image

load_dotenv()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(ROOT_DIR, "web-project")
THEME_DIR = os.path.join(ROOT_DIR, "factory-base-theme")
TOON_KEYS = ["slug", "title", "page_type", "description", "status"]
FORBIDDEN_SHARED_TAGS = ("nav", "header", "footer", "main")
DEFAULT_THEME = {
    "primary": "#1a365d",
    "secondary": "#2d3748",
    "accent": "#38b2ac",
    "font_url": "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap",
    "font_family": "'Plus Jakarta Sans', sans-serif",
}


def env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def strip_markdown(text: str) -> str:
    """Remove leading and trailing fenced code blocks from LLM output."""
    if not text:
        return ""
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def run_shell(command: str, cwd: str | None = None) -> str:
    if cwd is None:
        cwd = PROJECT_DIR
    print(f"Executing: {command}")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        return (result.stdout or "") + (result.stderr or "")
    except Exception as exc:
        return str(exc)


def git_push_permission_hint(push_output: str) -> str | None:
    lowered = push_output.lower()
    if "permission to " in lowered and " denied to " in lowered:
        return (
            "GitHub created the repository but denied git push. "
            "For a fine-grained PAT, add repository Contents: write in addition to "
            "Administration: write, then restart the app so the new token is loaded."
        )
    if "requested url returned error: 403" in lowered and "github.com" in lowered:
        return (
            "Git push was rejected with HTTP 403. "
            "This usually means the token can access repo metadata/admin operations but "
            "cannot write contents. Add repository Contents: write to GITHUB_TOKEN and restart the app."
        )
    return None


def encode_and_compress_image(file_path: str) -> str:
    with Image.open(file_path) as img:
        img = img.convert("RGB")
        img.thumbnail((1568, 1568), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_toon_to_json(toon_string: str) -> list[dict]:
    if not toon_string:
        return []

    cleaned = strip_markdown(toon_string)
    pages: list[dict] = []
    reader = csv.reader(StringIO(cleaned), delimiter="|", skipinitialspace=True)

    for row in reader:
        fields = [col.strip().strip('"').strip() for col in row if col.strip()]
        if not fields or fields[0].startswith("[+"):
            continue

        if len(fields) != len(TOON_KEYS):
            print(f"WARNING: TOON row has {len(fields)} fields, expected {len(TOON_KEYS)}: {fields}")
            fields = (fields + [""] * len(TOON_KEYS))[: len(TOON_KEYS)]

        page = dict(zip(TOON_KEYS, fields))
        page["slug"] = page["slug"].strip().lower()
        pages.append(page)

    return pages


def page_href(page: Mapping[str, Any]) -> str:
    slug = str(page.get("slug", "")).strip().lower()
    return "/" if slug == "index" else f"/{slug}"


def manifest_route_set(pages: list[dict]) -> set[str]:
    return {page_href(page) for page in pages}


def normalize_internal_href(href: str) -> str:
    cleaned = href.strip()
    if not cleaned:
        return "/"
    if "#" in cleaned:
        cleaned = cleaned.split("#", 1)[0]
    if "?" in cleaned:
        cleaned = cleaned.split("?", 1)[0]
    if cleaned != "/" and cleaned.endswith("/"):
        cleaned = cleaned.rstrip("/")
    return cleaned or "/"


def extract_xml_block(tag: str, content: str) -> str | None:
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    return strip_markdown(match.group(1).strip())


def build_architect_system_prompt() -> str:
    return """
You are a Senior Web Architect. Convert user requirements into three outputs:
1. a theme JSON object
2. a site_meta JSON object
3. a TOON page manifest

If a reference image is provided, extract the visual layout structure and encode it into the page descriptions as strict UI instructions.
Respect any explicit user copy, but always wrap it in concrete layout guidance.
The TOON description column may be multiline, but it must remain inside a quoted CSV cell.

Output ONLY these XML blocks. No extra commentary.

FORMAT:
<theme>
{"primary": "#HEX", "secondary": "#HEX", "accent": "#HEX", "font_url": "URL", "font_family": "FAMILY"}
</theme>
<site_meta>
{"brand_name": "BUSINESS NAME", "primary_cta_label": "OPTIONAL LABEL", "primary_cta_href": "OPTIONAL ROUTE"}
</site_meta>
<toon>
[+ slug, title, page_type, description, status]
| "index" | "Home" | "landing" | "Large hero with centered text and strong booking CTA." | "new" |
</toon>
""".strip()


def build_architect_user_text(state: "AgentState") -> str:
    return (
        f"Business Name: {state.get('business_name', '').strip()}\n"
        f"Job Type: {state.get('job_type', '').strip()}\n\n"
        "Requirements:\n"
        f"{state['requirements'].strip()}"
    )


def build_coder_system_prompt() -> str:
    return (
        "You are a Senior Web Developer. Build polished Astro page-body markup with Tailwind 4 and DaisyUI. "
        "Output CODE ONLY."
    )


def build_coder_user_prompt(
    page: dict,
    manifest_pages: list[str],
    site_meta: dict,
    assets: list[str],
    error_context: str,
) -> str:
    allowed_routes = ", ".join(manifest_pages)
    brand_name = site_meta.get("brand_name", "Brand")
    return (
        f"Write the Astro page BODY markup for '{page['title']}'. "
        f"Page type: {page['page_type']}. Description: {page['description']}. "
        f"Brand name for content reference only: {brand_name}. "
        "CRITICAL INSTRUCTIONS: "
        "- Output ONLY page-local content sections. "
        "- Do NOT generate <html>, <head>, <body>, <Layout>, <main>, <nav>, <header>, or <footer> tags. "
        "- The shared site shell is already handled by Layout.astro, so do NOT render site-wide navigation, footer links, or a global sticky header. "
        "- Wrap all major content sections in a centered container using `max-w-7xl mx-auto px-4 sm:px-6 lg:px-8`. "
        f"- Internal links may ONLY point to these routes: {allowed_routes}. You may also use `#...`, `mailto:`, `tel:`, or fully qualified external URLs. "
        "- For any page that is NOT the Home page, use a solid color top banner section with `bg-primary` or `bg-slate-900` and `text-white`. "
        "- For the Home page hero, use `bg-[url('/hero.jpg')]` with a dark `bg-slate-900/60` overlay for legibility. "
        "- For the Contact page map, embed a functional OpenStreetMap iframe or a polished styled placeholder with rounded corners. "
        "- FORM STRUCTURE RULE: For forms, you MUST wrap every input and textarea inside a standard DaisyUI `<label class=\"form-control w-full\">` wrapper. The text label must go inside a `<div class=\"label\"><span class=\"label-text\">` block above the input. Do not let labels float awkwardly. "
        "- FORM SPACING RULE: Forms MUST use clear vertical rhythm. The `<form>` element should use spacing such as `space-y-5` or `space-y-6`. Every input, date field, and textarea MUST include comfortable inner padding such as `px-4 py-3` or equivalent DaisyUI sizing so fields never look stuck together. Textareas should also have generous padding and a minimum height. The submit button must have visible separation from the last field with proper top spacing. "
        "- GRID ALIGNMENT RULE: When using CSS grids for side-by-side sections (like Contact Info next to a Form), you MUST apply `items-start` to the grid container to prevent the taller column from unnaturally stretching the shorter column. "
        "- NO DECORATIVE TEXT OVERLAPS: You are STRICTLY FORBIDDEN from using `absolute` positioning to place faded decorative text behind headings. Never overlap text on top of other text. "
        "- Headings (`h1`, `h2`, `h3`) should use `text-slate-900` or `text-primary`. "
        "- Add premium polish with DaisyUI cards, `shadow-xl`, `border border-slate-100`, and `rounded-2xl`. "
        "- When rendering checklist rows or icon-plus-text features, use `flex items-start gap-4`. "
        "- Any `https://placehold.co/...` image must include `w-full h-auto object-cover`. "
        f"- Available uploaded assets: {assets}. Never use `reference.jpg` in the generated page body. "
        "- Output code only with no markdown fences. "
        f"{error_context}"
    )


def normalize_site_meta(raw_site_meta: dict | None, business_name: str, pages: list[dict]) -> dict:
    site_meta = raw_site_meta if isinstance(raw_site_meta, dict) else {}
    allowed_routes = manifest_route_set(pages)
    business_name = business_name.strip()
    brand_name = business_name or str(site_meta.get("brand_name", "")).strip() or "Brand"

    primary_cta_label = site_meta.get("primary_cta_label")
    if not isinstance(primary_cta_label, str) or not primary_cta_label.strip():
        primary_cta_label = None
    else:
        primary_cta_label = primary_cta_label.strip()

    primary_cta_href = site_meta.get("primary_cta_href")
    if isinstance(primary_cta_href, str) and primary_cta_href.strip():
        normalized_href = normalize_internal_href(primary_cta_href.strip())
        if normalized_href in allowed_routes or normalized_href == "/":
            primary_cta_href = normalized_href
        else:
            primary_cta_href = None
    else:
        primary_cta_href = None

    return {
        "brand_name": brand_name,
        "primary_cta_label": primary_cta_label,
        "primary_cta_href": primary_cta_href,
    }


def validate_generated_page(raw_code: str, pages: list[dict]) -> list[str]:
    errors: list[str] = []
    allowed_routes = manifest_route_set(pages)
    lower_code = raw_code.lower()

    for tag in FORBIDDEN_SHARED_TAGS:
        if re.search(rf"<\s*{tag}\b", lower_code, re.IGNORECASE):
            errors.append(f"VALIDATION ERROR: generated page contains forbidden <{tag}> tag.")

    hrefs = re.findall(r"""href\s*=\s*["']([^"']+)["']""", raw_code, flags=re.IGNORECASE)
    seen_invalid_routes: set[str] = set()

    for href in hrefs:
        href = href.strip()
        if not href:
            continue
        if href.startswith(("#", "mailto:", "tel:", "http://", "https://", "//")):
            continue
        if not href.startswith("/"):
            continue

        normalized_href = normalize_internal_href(href)
        if normalized_href not in allowed_routes and normalized_href != "/":
            if normalized_href not in seen_invalid_routes:
                errors.append(
                    f"VALIDATION ERROR: generated page links to route '{href}' which is not present in manifest."
                )
                seen_invalid_routes.add(normalized_href)

    return errors


def write_theme_css(theme: dict) -> None:
    primary = theme.get("primary", DEFAULT_THEME["primary"])
    secondary = theme.get("secondary", DEFAULT_THEME["secondary"])
    accent = theme.get("accent", DEFAULT_THEME["accent"])
    font_url = theme.get("font_url", DEFAULT_THEME["font_url"])
    font_family = theme.get("font_family", DEFAULT_THEME["font_family"])

    css_content = f"""@import url('{font_url}');
@import "tailwindcss";
@plugin "daisyui";

@theme {{
  --color-primary: {primary};
  --color-secondary: {secondary};
  --color-accent: {accent};
  --font-sans: {font_family};
}}
"""
    styles_dir = os.path.join(PROJECT_DIR, "src", "styles")
    os.makedirs(styles_dir, exist_ok=True)
    css_path = os.path.join(styles_dir, "global.css")
    with open(css_path, "w") as handle:
        handle.write(css_content)


API_KEY = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=API_KEY)
print(f"OpenRouter API key configured: {bool(API_KEY)}")
ARCHITECT_MODEL = "anthropic/claude-sonnet-4"
CODER_MODEL = "anthropic/claude-sonnet-4"


class OrchestratorExecutionError(RuntimeError):
    def __init__(self, message: str, state: Mapping[str, Any]):
        super().__init__(message)
        self.state = dict(state)


class AgentState(TypedDict):
    project_id: str
    business_name: str
    job_type: str
    requirements: str
    assets: List[str]
    edit_mode: bool
    edit_prompt: str
    theme: dict
    site_meta: dict
    site_manifest: str
    build_logs: str
    build_errors: str
    iteration_count: int
    current_stage: str
    debug_logs: str
    error_message: str
    error_traceback: str
    provider_response_debug: str
    next: NotRequired[str]


# def append_debug_log(state: AgentState, message: str) -> None:
#     print(message)
#     existing = state.get("debug_logs", "")
#     state["debug_logs"] = f"{existing}\n{message}".strip() if existing else message
def append_debug_log(state: AgentState, message: str) -> None:
    print(message)
    existing = state.get("debug_logs", "")
    state["debug_logs"] = f"{existing}\n{message}".strip() if existing else message
    
    try:
        project_id = state.get("project_id", "unknown_project")
        safe_project_id = re.sub(r'[^a-zA-Z0-9_-]', '-', project_id).strip('-').lower()
        
        logs_dir = os.path.join(ROOT_DIR, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_file_path = os.path.join(logs_dir, f"{safe_project_id}.log")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"FAILED TO WRITE LOG TO DISK: {e}")

def set_stage(state: AgentState, stage: str, message: str) -> None:
    state["current_stage"] = stage
    append_debug_log(state, message)


def scrub_secret(text: str, secret: str | None) -> str:
    if not text or not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def clone_existing_project_from_github(state: AgentState) -> None:
    github_token = os.getenv("GITHUB_TOKEN")
    github_org = os.getenv("GITHUB_ORG_NAME")
    raw_repo_name = state.get("project_id", "generated-site")
    repo_name = re.sub(r"[^a-zA-Z0-9_-]", "-", raw_repo_name).strip("-").lower()

    if not repo_name:
        raise Exception("Edit mode requires a valid project_id to restore the GitHub repo.")
    if not github_token:
        raise Exception("Edit mode requires GITHUB_TOKEN to restore a missing local project from GitHub.")

    from github import Github, GithubException
    from github.AuthenticatedUser import AuthenticatedUser

    g = Github(github_token)
    try:
        if github_org:
            owner_name = github_org
            org = g.get_organization(github_org)
            org.get_repo(repo_name)
        else:
            user = cast(AuthenticatedUser, g.get_user())
            owner_name = user.login
            user.get_repo(repo_name)
    except GithubException as exc:
        owner_label = github_org or "authenticated user"
        raise Exception(f"Could not access GitHub repo '{repo_name}' under {owner_label}: {exc.data}") from exc

    remote_url = f"https://x-access-token:{github_token}@github.com/{owner_name}/{repo_name}.git"
    safe_remote_url = remote_url.replace(github_token, "[REDACTED]")
    append_debug_log(state, f"Verified GitHub repo exists: {owner_name}/{repo_name}")
    append_debug_log(state, f"Cloning {safe_remote_url} -> {PROJECT_DIR}")

    result = subprocess.run(
        ["git", "clone", remote_url, PROJECT_DIR],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR,
    )
    clone_output = scrub_secret((result.stdout or "") + (result.stderr or ""), github_token)
    if clone_output.strip():
        append_debug_log(state, f"Clone output:\n{clone_output}")
    if result.returncode != 0:
        raise Exception(f"Failed to clone GitHub repo {owner_name}/{repo_name}.")


def json_safe_dump(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [json_safe_dump(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe_dump(item) for key, item in value.items()}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
        except Exception:
            return repr(value)
    return repr(value)


def capture_provider_response_debug(response: Any) -> str:
    payload = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "usage": json_safe_dump(getattr(response, "usage", None)),
        "raw_response": json_safe_dump(response),
    }
    return json.dumps(payload, indent=2, default=str)


def extract_text_content(response: Any, actor_name: str) -> str:
    if response is None:
        raise ValueError(f"{actor_name} returned no response")

    choices = getattr(response, "choices", None)
    if choices is None:
        raise ValueError(f"{actor_name} returned no choices")
    if not choices:
        raise ValueError(f"{actor_name} returned an empty choices list")

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None:
        raise ValueError(f"{actor_name} returned no message")

    content = getattr(message, "content", None)
    if content is None:
        raise ValueError(f"{actor_name} returned no message content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                text_value = part.get("text")
            else:
                text_value = getattr(part, "text", None)
            if text_value:
                text_parts.append(str(text_value))
        if text_parts:
            return "\n".join(text_parts)

    raise ValueError(f"{actor_name} returned unsupported message content")


def raise_with_state(state: AgentState, exc: Exception) -> None:
    state["error_message"] = str(exc)
    state["error_traceback"] = traceback.format_exc()
    append_debug_log(state, f"ERROR [{state.get('current_stage', 'unknown')}]: {exc}")
    raise OrchestratorExecutionError(str(exc), state) from exc


def build_image_content_parts(state: AgentState) -> list[ChatCompletionContentPartParam]:
    image_parts: list[ChatCompletionContentPartParam] = []

    for asset_name in state.get("assets", []):
        lower_name = asset_name.lower()
        if not lower_name.endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        source_path = os.path.join(ROOT_DIR, asset_name)
        if not os.path.exists(source_path):
            append_debug_log(state, f"WARNING: Vision asset {asset_name} listed but not found on disk.")
            continue

        encoded_image = encode_and_compress_image(source_path)
        image_part: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        }
        image_parts.append(image_part)

    return image_parts


def bootstrapper_node(state: AgentState):
    set_stage(state, "bootstrapper", "\n--- AGENT: BOOTSTRAPPER IS SETTING UP ---")
    try:
        if state.get("edit_mode"):
            append_debug_log(state, f"EDIT MODE ENABLED. Bypassing bootstrap, loading existing project: {PROJECT_DIR}")
            if not os.path.exists(PROJECT_DIR):
                append_debug_log(state, f"Local project missing. Cloning {state['project_id']} from GitHub...")
                clone_existing_project_from_github(state)
                append_debug_log(state, "Installing dependencies for restored project...")
                append_debug_log(state, run_shell("npm i"))
            return state

        if os.path.exists(PROJECT_DIR):
            shutil.rmtree(PROJECT_DIR)
            append_debug_log(state, f"Deleted existing {PROJECT_DIR}")

        shutil.copytree(THEME_DIR, PROJECT_DIR)
        append_debug_log(state, f"Copied {THEME_DIR} -> {PROJECT_DIR}")

        public_dir = os.path.join(PROJECT_DIR, "public")
        os.makedirs(public_dir, exist_ok=True)

        for asset_name in state.get("assets", []):
            source_path = os.path.join(ROOT_DIR, asset_name)
            dest_path = os.path.join(public_dir, asset_name)
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                append_debug_log(state, f"Copied asset to public: {asset_name}")
            else:
                append_debug_log(state, f"WARNING: Asset {asset_name} listed but not found on disk.")

        write_theme_css(state.get("theme", DEFAULT_THEME))
        append_debug_log(state, "Wrote default global.css from bootstrap theme.")

        append_debug_log(state, "Installing dependencies...")
        append_debug_log(state, run_shell("npm i"))
        return state
    except Exception as exc:
        raise_with_state(state, exc)


def architect_node(state: AgentState):
    set_stage(state, "architect", "\n--- AGENT: ARCHITECT IS WORKING ---")
    response = None
    try:
        user_content: list[ChatCompletionContentPartParam] = build_image_content_parts(state)
        user_content.append(
            ChatCompletionContentPartTextParam(type="text", text=build_architect_user_text(state))
        )

        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": build_architect_system_prompt(),
        }
        user_message: ChatCompletionUserMessageParam = {"role": "user", "content": user_content}
        messages: list[ChatCompletionMessageParam] = [system_message, user_message]

        response = client.chat.completions.create(
            model=ARCHITECT_MODEL,
            messages=messages,
            max_tokens=8192,
            temperature=0.2,
        )
        state["provider_response_debug"] = capture_provider_response_debug(response)

        if response.usage:
            append_debug_log(
                state,
                (
                    "Architect Tokens -> "
                    f"prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens}, "
                    f"total: {response.usage.total_tokens}"
                ),
            )

        content = extract_text_content(response, "Architect")

        theme_payload = extract_xml_block("theme", content)
        extracted_theme = dict(DEFAULT_THEME)
        if theme_payload:
            try:
                parsed_theme = json.loads(theme_payload)
                if isinstance(parsed_theme, dict):
                    extracted_theme.update(parsed_theme)
                else:
                    append_debug_log(state, "WARNING: Parsed theme payload was not a JSON object. Using fallback theme.")
            except json.JSONDecodeError as exc:
                append_debug_log(state, f"WARNING: Failed to parse theme JSON: {exc}. Using fallback theme.")
        else:
            append_debug_log(state, "WARNING: No <theme> block found. Using fallback theme.")

        site_meta_payload = extract_xml_block("site_meta", content)
        raw_site_meta: dict = {}
        if site_meta_payload:
            try:
                parsed_site_meta = json.loads(site_meta_payload)
                if isinstance(parsed_site_meta, dict):
                    raw_site_meta = parsed_site_meta
                else:
                    append_debug_log(state, "WARNING: Parsed site_meta payload was not a JSON object. Using defaults.")
            except json.JSONDecodeError as exc:
                append_debug_log(state, f"WARNING: Failed to parse site_meta JSON: {exc}. Using defaults.")
        else:
            append_debug_log(state, "WARNING: No <site_meta> block found. Using defaults.")

        manifest = extract_xml_block("toon", content) or ""
        if not manifest:
            append_debug_log(state, "WARNING: No <toon> block found.")

        pages_data = parse_toon_to_json(manifest)
        normalized_site_meta = normalize_site_meta(raw_site_meta, state.get("business_name", ""), pages_data)

        write_theme_css(extracted_theme)
        append_debug_log(
            state,
            "Wrote global.css from extracted theme: "
            f"primary={extracted_theme.get('primary')}, "
            f"secondary={extracted_theme.get('secondary')}, "
            f"accent={extracted_theme.get('accent')}",
        )

        data_dir = os.path.join(PROJECT_DIR, "src", "data")
        os.makedirs(data_dir, exist_ok=True)
        json_path = os.path.join(data_dir, "pages.json")
        with open(json_path, "w") as handle:
            json.dump(pages_data, handle, indent=2)
        append_debug_log(state, f"Saved {len(pages_data)} pages to {json_path}")
        site_meta_path = os.path.join(data_dir, "site_meta.json")
        with open(site_meta_path, "w") as handle:
            json.dump(normalized_site_meta, handle, indent=2)
        append_debug_log(state, f"Saved site metadata to {site_meta_path}")

        append_debug_log(state, f"Generated Manifest:\n{manifest}")
        return {
            "theme": extracted_theme,
            "site_meta": normalized_site_meta,
            "site_manifest": manifest,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_stage": state.get("current_stage", ""),
            "debug_logs": state.get("debug_logs", ""),
            "provider_response_debug": state.get("provider_response_debug", ""),
        }
    except Exception as exc:
        if response is not None and not state.get("provider_response_debug"):
            state["provider_response_debug"] = capture_provider_response_debug(response)
        raise_with_state(state, exc)


def editor_node(state: AgentState):
    set_stage(state, "editor", "\n--- AGENT: EDITOR IS MODIFYING FILES ---")
    response = None
    try:
        edit_prompt = state.get("edit_prompt", "")
        if not edit_prompt:
            raise Exception("Edit mode triggered but no edit_prompt provided.")

        context_files = []
        pages_dir = os.path.join(PROJECT_DIR, "src", "pages")
        styles_dir = os.path.join(PROJECT_DIR, "src", "styles")

        if os.path.exists(pages_dir):
            for filename in os.listdir(pages_dir):
                if filename.endswith(".astro"):
                    filepath = os.path.join(pages_dir, filename)
                    with open(filepath, "r") as handle:
                        context_files.append(f"<file path=\"src/pages/{filename}\">\n{handle.read()}\n</file>")

        css_path = os.path.join(styles_dir, "global.css")
        if os.path.exists(css_path):
            with open(css_path, "r") as handle:
                context_files.append(f"<file path=\"src/styles/global.css\">\n{handle.read()}\n</file>")

        codebase_context = "\n".join(context_files)
        previous_errors = state.get("build_errors", "")
        error_context = ""
        if previous_errors:
            error_context = (
                "\n\nPREVIOUS BUILD FAILED:\n"
                f"{previous_errors}\n"
                "Fix these build errors while preserving the user's requested modification."
            )

        system_msg = (
            "You are an expert AI developer maintaining an Astro/Tailwind codebase. "
            "Review the user's requested change and the current codebase context. "
            "Output ONLY the fully updated code for the file(s) that need to be changed. "
            "You MUST wrap each modified file's code in XML tags containing the exact path, like this:\n"
            "<file path=\"src/pages/contact.astro\">\n...complete updated code...\n</file>\n"
            "Do NOT output markdown code blocks. Do not omit code for brevity. Provide the entire updated file."
        )

        user_msg = (
            f"CURRENT CODEBASE:\n{codebase_context}\n\n"
            f"USER REQUESTED MODIFICATION:\n{edit_prompt}"
            f"{error_context}"
        )

        response = client.chat.completions.create(
            model=CODER_MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            # max_tokens=8192,
            max_tokens=64000,
            temperature=0.1,
        )
        state["provider_response_debug"] = capture_provider_response_debug(response)

        if response.usage:
            append_debug_log(
                state,
                (
                    "Editor Tokens -> "
                    f"prompt: {response.usage.prompt_tokens}, "
                    f"completion: {response.usage.completion_tokens}, "
                    f"total: {response.usage.total_tokens}"
                ),
            )

        content = extract_text_content(response, "Editor")

        pattern = r'<file path="([^"]+)">\s*(.*?)\s*</file>'
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            append_debug_log(state, "WARNING: Editor did not output any valid <file> blocks. No changes made.")

        for path, new_code in matches:
            full_path = os.path.abspath(os.path.join(PROJECT_DIR, path))
            project_root = os.path.abspath(PROJECT_DIR)
            if not full_path.startswith(project_root + os.sep):
                append_debug_log(state, f"WARNING: Editor tried to modify {path} outside the project. Skipping.")
                continue
            if os.path.exists(full_path):
                with open(full_path, "w") as handle:
                    handle.write(new_code.strip())
                append_debug_log(state, f"SUCCESS: Edited {path}")
            else:
                append_debug_log(state, f"WARNING: Editor tried to modify {path} which does not exist locally.")

        append_debug_log(state, "Running Build Test after edits...")
        build_output = run_shell("npm run build")
        append_debug_log(state, f"Build output:\n{build_output}")

        return {
            "build_logs": build_output,
            "iteration_count": state.get("iteration_count", 0) + 1,
            "current_stage": state.get("current_stage", ""),
            "debug_logs": state.get("debug_logs", ""),
            "provider_response_debug": state.get("provider_response_debug", ""),
        }

    except Exception as exc:
        if response is not None and not state.get("provider_response_debug"):
            state["provider_response_debug"] = capture_provider_response_debug(response)
        raise_with_state(state, exc)


def coder_node(state: AgentState):
    set_stage(state, "coder", "\n--- AGENT: CODER IS GENERATING STATIC PAGES ---")
    try:
        pages_dir = os.path.join(PROJECT_DIR, "src", "pages")
        json_path = os.path.join(PROJECT_DIR, "src", "data", "pages.json")
        os.makedirs(pages_dir, exist_ok=True)

        with open(json_path, "r") as handle:
            pages = json.load(handle)

        append_debug_log(state, f"Generating {len(pages)} static pages...")
        errors: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        manifest_pages = sorted(manifest_route_set(pages))

        for page in pages:
            slug = page["slug"]
            title = page["title"]
            append_debug_log(state, f"  -> Generating {slug}.astro ...")

            prev_errors = state.get("build_errors", "")
            error_context = ""
            if prev_errors:
                error_context = f"PREVIOUS BUILD OR VALIDATION FAILED:\n{prev_errors}\nFix those issues in this page."

            user_prompt = build_coder_user_prompt(
                page=page,
                manifest_pages=manifest_pages,
                site_meta=state.get("site_meta", {}),
                assets=state.get("assets", []),
                error_context=error_context,
            )
            coder_user_content: list[ChatCompletionContentPartParam] = build_image_content_parts(state)
            coder_user_content.append(ChatCompletionContentPartTextParam(type="text", text=user_prompt))

            try:
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": build_coder_system_prompt(),
                }
                user_message: ChatCompletionUserMessageParam = {"role": "user", "content": coder_user_content}
                messages: list[ChatCompletionMessageParam] = [system_message, user_message]

                response = client.chat.completions.create(
                    model=CODER_MODEL,
                    messages=messages,
                    # max_tokens=8192,
                    max_tokens=64000,
                    temperature=0.1,
                )
                state["provider_response_debug"] = capture_provider_response_debug(response)

                if response.usage:
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                    append_debug_log(
                        state,
                        f"     Tokens -> prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}",
                    )

                raw_code = strip_markdown(extract_text_content(response, "Coder"))
                validation_errors = validate_generated_page(raw_code, pages)
                if validation_errors:
                    for validation_error in validation_errors:
                        append_debug_log(state, f"     {validation_error}")
                    errors.extend(validation_errors)

                astro_code = (
                    "---\n"
                    "import Layout from '../layouts/Layout.astro';\n"
                    "---\n"
                    f"<Layout title=\"{title}\">\n{raw_code}\n</Layout>"
                )
                file_path = os.path.join(pages_dir, f"{slug}.astro")
                with open(file_path, "w") as handle:
                    handle.write(astro_code)
                append_debug_log(state, f"     Saved {file_path}")

            except Exception as exc:
                error_msg = f"GENERATION ERROR: failed to generate {slug}.astro: {exc}"
                append_debug_log(state, f"     {error_msg}")
                errors.append(error_msg)

        total = total_prompt_tokens + total_completion_tokens
        append_debug_log(
            state,
            f"Coder Totals -> prompt: {total_prompt_tokens}, completion: {total_completion_tokens}, total: {total}",
        )

        append_debug_log(state, "Running Build Test...")
        build_output = run_shell("npm run build")
        append_debug_log(state, f"Build output:\n{build_output}")

        if errors:
            build_output = "\n".join(errors + [build_output]).strip()

        return {
            "build_logs": build_output,
            "iteration_count": state["iteration_count"] + 1,
            "current_stage": state.get("current_stage", ""),
            "debug_logs": state.get("debug_logs", ""),
            "provider_response_debug": state.get("provider_response_debug", ""),
        }
    except Exception as exc:
        raise_with_state(state, exc)


def reviewer_node(state: AgentState):
    set_stage(state, "reviewer", "\n--- AGENT: REVIEWER IS CHECKING ---")
    try:
        build_logs = state["build_logs"] or ""
        deploy_enabled = env_flag("ENABLE_DEPLOY", default=False)
        validation_or_generation_lines = [
            line
            for line in build_logs.splitlines()
            if "VALIDATION ERROR" in line or "GENERATION ERROR" in line
        ]

        if validation_or_generation_lines:
            append_debug_log(state, f"VALIDATION OR GENERATION FAILED. Iteration: {state['iteration_count']}")
            if state["iteration_count"] > 3:
                append_debug_log(state, "Max retries reached. Stopping.")
                return {
                    "next": END,
                    "build_errors": "\n".join(validation_or_generation_lines),
                    "current_stage": state.get("current_stage", ""),
                    "debug_logs": state.get("debug_logs", ""),
                }

            build_errors = "\n".join(validation_or_generation_lines)
            retry_target = "editor" if state.get("edit_mode") else "coder"
            append_debug_log(state, f"Feeding back to {retry_target}: {build_errors}")
            return {
                "next": retry_target,
                "build_errors": build_errors,
                "current_stage": state.get("current_stage", ""),
                "debug_logs": state.get("debug_logs", ""),
            }

        if "Complete!" in build_logs:
            next_step = "deployer" if deploy_enabled else END
            if deploy_enabled:
                append_debug_log(state, "BUILD SUCCESSFUL. Routing to deployer.")
            else:
                append_debug_log(state, "BUILD SUCCESSFUL. Exiting without deploy.")
            return {
                "next": next_step,
                "build_errors": "",
                "current_stage": state.get("current_stage", ""),
                "debug_logs": state.get("debug_logs", ""),
            }

        append_debug_log(state, f"BUILD FAILED. Iteration: {state['iteration_count']}")
        if state["iteration_count"] > 3:
            append_debug_log(state, "Max retries reached. Stopping.")
            return {
                "next": END,
                "build_errors": "",
                "current_stage": state.get("current_stage", ""),
                "debug_logs": state.get("debug_logs", ""),
            }

        error_lines = [
            line
            for line in build_logs.splitlines()
            if "ERROR" in line or "Could not resolve" in line or "error" in line.lower()
        ]
        build_errors = "\n".join(error_lines) if error_lines else "Unknown build error"
        retry_target = "editor" if state.get("edit_mode") else "coder"
        append_debug_log(state, f"Feeding back to {retry_target}: {build_errors}")
        return {
            "next": retry_target,
            "build_errors": build_errors,
            "current_stage": state.get("current_stage", ""),
            "debug_logs": state.get("debug_logs", ""),
        }
    except Exception as exc:
        raise_with_state(state, exc)


# def deployer_node(state: AgentState):
#     set_stage(state, "deployer", "\n--- AGENT: DEPLOYER IS PUBLISHING ---")
#     try:
#         github_token = os.getenv("GITHUB_TOKEN")
#         github_org = os.getenv("GITHUB_ORG_NAME")
#         repo_name = state["project_id"]

#         if not github_token:
#             append_debug_log(state, "SKIPPED: No GITHUB_TOKEN set in .env")
#             return state

#         owner = github_org if github_org else run_shell("gh api user --jq '.login'").strip()
#         remote_url = f"https://x-access-token:{github_token}@github.com/{owner}/{repo_name}.git"

#         if github_org:
#             append_debug_log(state, f"Creating repo under org: {github_org}/{repo_name}")
#             run_shell(f"gh repo create {github_org}/{repo_name} --public --confirm", cwd=PROJECT_DIR)
#         else:
#             append_debug_log(state, f"Creating personal repo: {repo_name}")
#             run_shell(f"gh repo create {repo_name} --public --confirm", cwd=PROJECT_DIR)

#         run_shell("git init", cwd=PROJECT_DIR)
#         run_shell("git add -A", cwd=PROJECT_DIR)
#         run_shell('git commit -m "Initial site generated by Site Factory"', cwd=PROJECT_DIR)
#         run_shell("git branch -M main", cwd=PROJECT_DIR)
#         run_shell(f"git remote add origin {remote_url}", cwd=PROJECT_DIR)
#         push_output = run_shell("git push -u origin main", cwd=PROJECT_DIR)
#         append_debug_log(state, f"Push output:\n{push_output}")
#         append_debug_log(state, f"Deployed to: https://github.com/{owner}/{repo_name}")
#         return state
#     except Exception as exc:
#         raise_with_state(state, exc)
def deployer_node(state: AgentState):
    set_stage(state, "deployer", "\n--- AGENT: DEPLOYER IS PUBLISHING ---")
    try:
        github_token = os.getenv("GITHUB_TOKEN")
        github_org = os.getenv("GITHUB_ORG_NAME")
        NETLIFY_AUTH_TOKEN = os.getenv("NETLIFY_AUTH_TOKEN")
        
        # Sanitize the project ID to ensure it is a valid GitHub repo name
        raw_repo_name = state.get("project_id", "generated-site")
        repo_name = re.sub(r'[^a-zA-Z0-9_-]', '-', raw_repo_name).strip('-').lower()

        if not github_token:
            append_debug_log(state, "SKIPPED: No GITHUB_TOKEN set in .env")
            return state

        from github import Github, GithubException
        from github.AuthenticatedUser import AuthenticatedUser
        g = Github(github_token)

        # 1. Determine Owner and Create Repo via PyGithub
        try:
            if github_org:
                org = g.get_organization(github_org)
                owner_name = github_org
                append_debug_log(state, f"Attempting to create repo: {owner_name}/{repo_name}")
                repo = org.create_repo(name=repo_name, private=False)
            else:
                user = cast(AuthenticatedUser, g.get_user())
                owner_name = user.login
                append_debug_log(state, f"Attempting to create repo: {owner_name}/{repo_name}")
                repo = user.create_repo(name=repo_name, private=False)
            append_debug_log(state, "Repo created successfully via PyGithub.")
            
        except GithubException as e:
            if e.status == 422 and "name already exists" in str(e.data):
                append_debug_log(state, f"Repo {owner_name}/{repo_name} already exists. Will push updates.")
            else:
                raise Exception(f"Failed to create repo via PyGithub: {e.data}")

        # 2. Setup Git and Push
        remote_url = f"https://x-access-token:{github_token}@github.com/{owner_name}/{repo_name}.git"

        # Wipe existing local .git folder to ensure a clean slate if regenerating
        git_dir = os.path.join(PROJECT_DIR, ".git")
        if os.path.exists(git_dir):
            shutil.rmtree(git_dir)

        run_shell("git init", cwd=PROJECT_DIR)
        
        # Configure local bot identity so cloud commits don't fail
        run_shell('git config user.name "Site Factory Bot"', cwd=PROJECT_DIR)
        run_shell('git config user.email "bot@sitefactory.local"', cwd=PROJECT_DIR)
        
        run_shell("git add -A", cwd=PROJECT_DIR)
        run_shell('git commit -m "Auto-generated site payload"', cwd=PROJECT_DIR)
        run_shell("git branch -M main", cwd=PROJECT_DIR)
        run_shell(f"git remote add origin {remote_url}", cwd=PROJECT_DIR)
        
        # Force push guarantees the remote matches the new generated output exactly
        push_output = run_shell("git push -u origin main --force", cwd=PROJECT_DIR)
        push_hint = git_push_permission_hint(push_output)
        
        append_debug_log(state, f"Push output:\n{push_output}")
        if push_hint:
            append_debug_log(state, push_hint)
            raise Exception(push_hint)

        import tempfile
        import time
        import urllib.request
        import zipfile

        if NETLIFY_AUTH_TOKEN:
            dist_dir = os.path.join(PROJECT_DIR, "dist")
            if not os.path.isdir(dist_dir):
                raise Exception(f"Netlify deploy failed: built output directory not found at {dist_dir}")

            append_debug_log(state, "Uploading built site to Netlify...")
            netlify_api_url = "https://api.netlify.com/api/v1/sites"
            netlify_headers = {
                "Authorization": f"Bearer {NETLIFY_AUTH_TOKEN}",
                "Content-Type": "application/json"
            }
            netlify_payload = json.dumps({
                "name": f"{repo_name}-{os.urandom(4).hex()}"
            }).encode("utf-8")
            
            netlify_req = urllib.request.Request(netlify_api_url, data=netlify_payload, headers=netlify_headers)
            try:
                with urllib.request.urlopen(netlify_req) as response:
                    site_data = json.loads(response.read().decode())
                    site_id = site_data.get("id")
                    site_admin_url = site_data.get("admin_url")
                    if site_admin_url:
                        append_debug_log(state, f"Netlify site created: {site_admin_url}")

                    if not site_id:
                        raise Exception("Netlify created a site but did not return a site ID.")

                    dist_file_count = 0
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temp_zip_file:
                        zip_path = temp_zip_file.name

                    try:
                        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                            for root, _, files in os.walk(dist_dir):
                                for filename in files:
                                    file_path = os.path.join(root, filename)
                                    arcname = os.path.relpath(file_path, dist_dir)
                                    zip_file.write(file_path, arcname)
                                    dist_file_count += 1

                        if dist_file_count == 0:
                            raise Exception(f"Netlify deploy failed: no built files found in {dist_dir}")

                        with open(zip_path, "rb") as zip_stream:
                            deploy_payload = zip_stream.read()

                        deploy_headers = {
                            "Authorization": f"Bearer {NETLIFY_AUTH_TOKEN}",
                            "Content-Type": "application/zip"
                        }
                        deploy_api_url = f"https://api.netlify.com/api/v1/sites/{site_id}/deploys"
                        deploy_req = urllib.request.Request(deploy_api_url, data=deploy_payload, headers=deploy_headers)

                        with urllib.request.urlopen(deploy_req) as deploy_response:
                            deploy_data = json.loads(deploy_response.read().decode())
                    finally:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)

                    deploy_id = deploy_data.get("id")
                    if not deploy_id:
                        raise Exception("Netlify did not return a deploy ID after uploading the built site.")

                    append_debug_log(state, "Waiting for Netlify deploy to finish...")
                    live_url = None
                    deploy_status_url = f"https://api.netlify.com/api/v1/deploys/{deploy_id}"

                    for attempt in range(1, 13):
                        status_req = urllib.request.Request(deploy_status_url, headers={"Authorization": f"Bearer {NETLIFY_AUTH_TOKEN}"})
                        with urllib.request.urlopen(status_req) as status_response:
                            latest_deploy = json.loads(status_response.read().decode())

                        deploy_state = str(latest_deploy.get("state", "")).lower()
                        deploy_error = (latest_deploy.get("error_message") or "").strip()

                        if deploy_error or deploy_state in {"error", "failed"}:
                            raise Exception(
                                f"Netlify deploy failed: {deploy_error or f'state={deploy_state}'}"
                            )

                        if deploy_state in {"ready", "current"}:
                            live_url = (
                                latest_deploy.get("deploy_ssl_url")
                                or latest_deploy.get("ssl_url")
                                or latest_deploy.get("deploy_url")
                                or site_data.get("ssl_url")
                                or site_data.get("url")
                            )
                            append_debug_log(state, f"SUCCESS! Live Preview URL: {live_url}")
                            break

                        append_debug_log(
                            state,
                            f"Netlify deploy status: {deploy_state or 'unknown'} (attempt {attempt}/12).",
                        )
                        time.sleep(5)

                    if not live_url:
                        raise Exception("Netlify deploy did not become ready before the polling timeout.")
            except Exception as e:
                raise Exception(f"GitHub push succeeded, but Netlify deploy failed: {e}")
        else:
            append_debug_log(state, "No NETLIFY_AUTH_TOKEN found. Skipping auto-deploy.")
        append_debug_log(state, f"Deployed to: https://github.com/{owner_name}/{repo_name}")
        return state

    except Exception as exc:
        raise_with_state(state, exc)

workflow = StateGraph(AgentState)
workflow.add_node("bootstrapper", bootstrapper_node)
workflow.add_node("architect", architect_node)
workflow.add_node("coder", coder_node)
workflow.add_node("editor", editor_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("deployer", deployer_node)

workflow.set_entry_point("bootstrapper")
workflow.add_conditional_edges(
    "bootstrapper",
    lambda state: "editor" if state.get("edit_mode") else "architect",
    {
        "editor": "editor",
        "architect": "architect",
    },
)
workflow.add_edge("architect", "coder")
workflow.add_edge("coder", "reviewer")
workflow.add_edge("editor", "reviewer")
workflow.add_conditional_edges(
    "reviewer",
    lambda state: state["next"],
    {
        "coder": "coder",
        "editor": "editor",
        "deployer": "deployer",
        END: END,
    },
)
workflow.add_edge("deployer", END)

factory = workflow.compile()
print("Factory pipeline initialized v12")


# if __name__ == "__main__":
#     test_input: AgentState = {
#         "project_id": "smile-clinic-01",
#         "business_name": "Apex Dental Care",
#         "job_type": "NEW",
#         "requirements": """
#         I need a premium 5-page website for 'Apex Dental Care'. The tone must be high-end, modern, and deeply trustworthy.
#         Focus heavily on patient comfort and advanced technology.
#         Pages required:
#         1. Home: High-conversion landing page focused on booking appointments.
#         2. Services: Detailed breakdown focusing on Cosmetic Dentistry, Implants, and Invisalign.
#         3. Our Team: Highlight the expertise of the head surgeons.
#         4. Patient Experience: Information on what to expect, insurance, and financing.
#         5. Contact: Form, map, and emergency dental hotlines.
#         Write persuasive, conversion-focused copywriting. Do not use generic 'lorem ipsum'.
#         """,
#         "assets": ["reference.jpg", "hero.jpg"],
#         "theme": {},
#         "site_meta": {},
#         "site_manifest": "",
#         "build_logs": "",
#         "build_errors": "",
#         "iteration_count": 0,
#         "current_stage": "",
#         "debug_logs": "",
#         "error_message": "",
#         "error_traceback": "",
#         "provider_response_debug": "",
#     }

#     print("Starting the Factory...")
#     try:
#         final_state = factory.invoke(test_input)
#     except OrchestratorExecutionError as exc:
#         print("\n--- ORCHESTRATOR FAILURE ---")
#         print(exc.state.get("error_traceback", str(exc)))
#         if exc.state.get("provider_response_debug"):
#             print("\n--- PROVIDER RESPONSE DEBUG ---")
#             print(exc.state["provider_response_debug"])
#         raise

#     print("\n--- FINAL TOON MANIFEST ---")
#     print(final_state["site_manifest"])
