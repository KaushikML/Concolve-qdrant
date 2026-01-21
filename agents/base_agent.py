from typing import Any, Dict, List, Optional


class BaseAgent:
    name = "base"

    def run(
        self,
        source_ids: Optional[List[str]] = None,
        force_full_scan: bool = False,
        run_decay: bool = False,
    ) -> Dict[str, Any]:
        raise NotImplementedError
