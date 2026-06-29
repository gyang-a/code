from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from code_agent.services.persistence import list_sessions, record_session


class PersistenceTests(unittest.TestCase):
    def test_record_session_preserves_long_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            title = "项目名称：" + "个人备忘录管理系统" * 50

            record_session(workspace, "thread_1", title=title)

            records = list_sessions(workspace)
            self.assertEqual(records[0].title, title)


if __name__ == "__main__":
    unittest.main()
