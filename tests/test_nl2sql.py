import importlib
import sys
import types
import unittest


class NL2SQLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault(
            "oracledb",
            types.SimpleNamespace(Error=Exception, connect=lambda *args, **kwargs: None),
        )
        if "requests" not in sys.modules:
            class _DummyResponse:
                def raise_for_status(self):
                    return None

                def json(self):  # pragma: no cover - não usado nos testes
                    return {}

            sys.modules["requests"] = types.SimpleNamespace(post=lambda *args, **kwargs: _DummyResponse())
        cls.nl2sql = importlib.import_module("nl2sql")

    def test_tokenize_handles_accents_and_short_words(self):
        tokens = self.nl2sql.tokenize("A sessão está muito lenta no servidor Oracle!")
        self.assertIn("sessão", tokens)
        self.assertIn("lenta", tokens)
        self.assertNotIn("ora", tokens)

    def test_is_safe_select_blocks_dml_and_requires_select(self):
        self.assertFalse(self.nl2sql.is_safe_select("DELETE FROM tabela"))
        self.assertFalse(self.nl2sql.is_safe_select("DROP TABLE X"))
        self.assertTrue(self.nl2sql.is_safe_select("SELECT * FROM tabela"))

    def test_extract_tables_and_infra_only_validation(self):
        sql = """
        SELECT * FROM dba_users du
        JOIN gv$session s ON du.username = s.username
        """
        tables = self.nl2sql.extract_tables_from_sql(sql)
        self.assertIn("DBA_USERS", tables)
        self.assertIn("GV$SESSION", tables)
        self.assertTrue(self.nl2sql.is_infra_sql_only(sql))

        invalid_sql = "SELECT * FROM app.users"
        self.assertFalse(self.nl2sql.is_infra_sql_only(invalid_sql))

    def test_metadata_cache_finds_best_table(self):
        cache = self.nl2sql.MetadataCache()
        cache.add_table("APP", "CLIENTES", "Tabela de clientes")
        cache.add_column("APP", "CLIENTES", "ID", "NUMBER", 10, "N", "Identificador")
        cache.add_column("APP", "CLIENTES", "NOME", "VARCHAR2", 50, "Y", "Nome do cliente")

        best = cache.find_best_table(["cliente", "nome"])
        self.assertIsNotNone(best)
        _, table_meta, score = best
        self.assertEqual(table_meta.name, "CLIENTES")
        self.assertGreater(score, 0)

    def test_build_sql_from_business_plan_filters_invalid_entries(self):
        cache = self.nl2sql.MetadataCache()
        cache.add_table("APP", "PEDIDOS", "Pedidos de venda")
        cache.add_column("APP", "PEDIDOS", "ID_PEDIDO", "NUMBER", 10, "N", "")
        cache.add_column("APP", "PEDIDOS", "DATA_PEDIDO", "DATE", None, "Y", "")
        cache.add_column("APP", "PEDIDOS", "VALOR_TOTAL", "NUMBER", 12, "Y", "")

        plan = {
            "select_columns": ["id_pedido", "valor_total", "inexistente"],
            "where_clauses": ["DATA_PEDIDO >= TRUNC(SYSDATE)", "COLUNA_FAKE = 1"],
        }

        sql = self.nl2sql.build_sql_from_business_plan(cache.tables[cache.key("APP", "PEDIDOS")], plan)
        self.assertIn("ID_PEDIDO", sql)
        self.assertIn("VALOR_TOTAL", sql)
        self.assertIn("DATA_PEDIDO", sql)
        self.assertNotIn("COLUNA_FAKE", sql)

    def test_strip_sql_from_llm_extracts_code_blocks(self):
        text = """
        Aqui está o SQL:
        ```sql
        SELECT * FROM dual;
        ```
        Obrigado
        """
        result = self.nl2sql.strip_sql_from_llm(text)
        self.assertEqual("SELECT * FROM dual;", result)

    def test_parse_llm_json_handles_wrapped_text(self):
        raw = "Resposta:\n{" "\"select_columns\": [\"COL1\"], \"where_clauses\": []" "}"
        parsed = self.nl2sql.parse_llm_json(raw)
        self.assertIsNotNone(parsed)
        self.assertEqual(["COL1"], parsed["select_columns"])


if __name__ == "__main__":
    unittest.main()
