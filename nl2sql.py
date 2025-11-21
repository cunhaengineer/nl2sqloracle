#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NL2SQL Oracle com:
- Metadados de negócio (DBA_TABLES / DBA_TAB_COLUMNS / DBA_*_COMMENTS)
- LLM (Ollama ou OpenAI)
- Modo NEGÓCIO (tabelas da aplicação)
- Modo INFRA (DBA_*, V$, GV$) genérico

Fluxo:
- Usuário faz pergunta em linguagem natural.
- Script sugere se a pergunta é INFRA ou NEGÓCIO, mas o usuário escolhe.
- Gera SQL apenas SELECT/WITH em dialeto Oracle.
- NEGÓCIO: usa dicionário de metadados + LLM para montar SELECT em uma tabela.
- INFRA: usa LLM para gerar SELECT em views DBA_*/V$/GV$/CDB_ com whitelist + validação.
- Depois de gerar o SQL, pergunta que tipo de saída o usuário quer:
  [1] primeiras linhas, [2] COUNT(*), [3] ambos.

Requisitos:
    pip install oracledb requests openai (se usar OpenAI)
Variáveis de ambiente:
    ORA_USER, ORA_PASSWORD, ORA_DSN

    NL2SQL_PROVIDER = "ollama" ou "openai" (default: "ollama")

    # Ollama
    OLLAMA_BASE_URL (default: http://localhost:11434)
    OLLAMA_MODEL    (default: llama3.1)

    # OpenAI ou compatível
    OPENAI_API_KEY
    OPENAI_MODEL   (default: gpt-4.1-mini)
    OPENAI_BASE_URL (opcional)
"""

import os
import re
import sys
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import oracledb
import requests


# ==========================
# Config LLM
# ==========================

def get_llm_provider() -> str:
    return os.getenv("NL2SQL_PROVIDER", "ollama").lower()


def call_ollama(prompt: str) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    url = f"{base_url.rstrip('/')}/api/chat"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Você gera exclusivamente SQL para Oracle Database (apenas SELECT/WITH), "
                    "usando o dialeto Oracle (funções e sintaxe Oracle). "
                    "Nunca use DML/DDL (INSERT/UPDATE/DELETE/MERGE, CREATE, ALTER, DROP, etc.)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["message"]["content"]


def call_openai(prompt: str) -> str:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("Pacote 'openai' não instalado. pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY não definido no ambiente.")

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    base_url = os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Você gera exclusivamente SQL para Oracle Database (apenas SELECT/WITH), "
                    "usando o dialeto Oracle (funções e sintaxe Oracle). "
                    "Nunca use DML/DDL (INSERT/UPDATE/DELETE/MERGE, CREATE, ALTER, DROP, etc.)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def call_llm(prompt: str) -> str:
    provider = get_llm_provider()
    try:
        if provider == "ollama":
            return call_ollama(prompt)
        if provider == "openai":
            return call_openai(prompt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Falha ao chamar provider {provider}: {exc}") from exc

    raise RuntimeError(f"NL2SQL_PROVIDER não suportado: {provider}")


def validate_provider_configuration() -> None:
    """Valida configurações mínimas do provedor escolhido."""
    provider = get_llm_provider()

    if provider == "ollama":
        if not os.getenv("OLLAMA_MODEL", "").strip():
            raise RuntimeError("OLLAMA_MODEL não definido para o provedor Ollama.")
        return

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY não definido para o provedor OpenAI.")
        if not os.getenv("OPENAI_MODEL", "").strip():
            raise RuntimeError("OPENAI_MODEL não definido para o provedor OpenAI.")
        return

    raise RuntimeError(f"NL2SQL_PROVIDER não suportado: {provider}")


# ==========================
# Metadados de NEGÓCIO
# ==========================

@dataclass
class ColumnMeta:
    name: str
    data_type: str
    data_length: Optional[int]
    nullable: bool
    comment: Optional[str] = None


@dataclass
class TableMeta:
    owner: str
    name: str
    comment: Optional[str] = None
    columns: Dict[str, ColumnMeta] = field(default_factory=dict)

    def search_text(self) -> str:
        parts: List[str] = [
            self.owner,
            self.name,
            self.comment or "",
        ]
        for col in self.columns.values():
            parts.append(col.name)
            if col.comment:
                parts.append(col.comment)
        return " ".join(parts).lower()


@dataclass
class MetadataCache:
    tables: Dict[str, TableMeta] = field(default_factory=dict)

    def key(self, owner: str, table_name: str) -> str:
        return f"{owner.upper()}.{table_name.upper()}"

    def add_table(self, owner: str, table_name: str, comment: Optional[str]):
        k = self.key(owner, table_name)
        if k not in self.tables:
            self.tables[k] = TableMeta(
                owner=owner.upper(),
                name=table_name.upper(),
                comment=comment,
            )
        else:
            if comment and not self.tables[k].comment:
                self.tables[k].comment = comment

    def add_column(
        self,
        owner: str,
        table_name: str,
        column_name: str,
        data_type: str,
        data_length: Optional[int],
        nullable: str,
        comment: Optional[str],
    ):
        k = self.key(owner, table_name)
        if k not in self.tables:
            self.tables[k] = TableMeta(
                owner=owner.upper(),
                name=table_name.upper(),
                comment=None,
            )
        table = self.tables[k]
        col_name = column_name.upper()
        table.columns[col_name] = ColumnMeta(
            name=col_name,
            data_type=data_type.upper(),
            data_length=data_length,
            nullable=(nullable == "Y"),
            comment=comment,
        )

    def find_best_table(self, tokens: List[str]) -> Optional[Tuple[str, TableMeta, int]]:
        if not tokens:
            return None

        best_key = None
        best_score = 0
        best_table = None

        for k, t in self.tables.items():
            text = t.search_text()
            score = 0
            for tok in tokens:
                if tok in text:
                    score += 1
            if score > best_score:
                best_score = score
                best_key = k
                best_table = t

        if best_key is None or best_score == 0:
            return None
        return best_key, best_table, best_score


# ==========================
# Conexão e metadados Oracle
# ==========================

def get_connection():
    user = os.getenv("ORA_USER", "USER_AJUSTE")
    password = os.getenv("ORA_PASSWORD", "SENHA_AJUSTE")
    dsn = os.getenv("ORA_DSN", "host:1521/servico")

    try:
        conn = oracledb.connect(user=user, password=password, dsn=dsn)
        return conn
    except oracledb.Error as e:
        print("Erro ao conectar no Oracle:", e)
        sys.exit(1)


def load_metadata(conn, owners_filter: Optional[List[str]] = None) -> MetadataCache:
    cache = MetadataCache()
    owners_filter = [o.upper() for o in owners_filter] if owners_filter else None

    where_owners = ""
    bind_owners = []
    if owners_filter:
        where_owners = " AND t.owner IN ({})".format(
            ",".join([f":own{i}" for i in range(len(owners_filter))])
        )
        bind_owners = owners_filter

    sql_tables = f"""
        SELECT
            t.owner,
            t.table_name,
            c.comments
        FROM
            dba_tables t
            LEFT JOIN dba_tab_comments c
                ON c.owner = t.owner
               AND c.table_name = t.table_name
        WHERE
            t.owner NOT IN ('SYS','SYSTEM')
            {where_owners}
    """

    with conn.cursor() as cur:
        cur.execute(sql_tables, bind_owners)
        for owner, table_name, comment in cur:
            cache.add_table(owner, table_name, comment)

    where_owners_col = ""
    bind_owners_col = []
    if owners_filter:
        where_owners_col = " AND c.owner IN ({})".format(
            ",".join([f":own{i}" for i in range(len(owners_filter))])
        )
        bind_owners_col = owners_filter

    sql_cols = f"""
        SELECT
            c.owner,
            c.table_name,
            c.column_name,
            c.data_type,
            c.data_length,
            c.nullable,
            cc.comments
        FROM
            dba_tab_columns c
            LEFT JOIN dba_col_comments cc
                ON cc.owner = c.owner
               AND cc.table_name = c.table_name
               AND cc.column_name = c.column_name
        WHERE
            c.owner NOT IN ('SYS','SYSTEM')
            {where_owners_col}
    """

    with conn.cursor() as cur:
        cur.execute(sql_cols, bind_owners_col)
        for owner, table_name, col_name, dt, length, nullable, comment in cur:
            cache.add_column(
                owner=owner,
                table_name=table_name,
                column_name=col_name,
                data_type=dt,
                data_length=length,
                nullable=nullable,
                comment=comment,
            )

    print(f"Metadados carregados: {len(cache.tables)} tabelas.")
    return cache


# ==========================
# Utilidades de texto / segurança
# ==========================

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^\wçáéíóúãõâêôü]+", " ", text, flags=re.UNICODE)
    tokens = [t for t in text.split() if len(t) > 2]
    return tokens


def is_safe_select(sql: str) -> bool:
    sql_clean = sql.strip().strip(";")
    upper = sql_clean.upper()

    forbidden = [
        "INSERT ", "UPDATE ", "DELETE ", "MERGE ", "DROP ", "ALTER ",
        "TRUNCATE ", "CREATE ", "GRANT ", "REVOKE "
    ]
    for kw in forbidden:
        if kw in upper:
            return False

    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False

    return True


def strip_sql_from_llm(text: str) -> str:
    # pega bloco ```sql ... ``` se existir
    code_block = re.search(r"```(?:sql)?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_block:
        sql = code_block.group(1)
    else:
        sql = text
    return sql.strip()


# ==========================
# MODO INFRA: detecção + LLM genérico
# ==========================

# Whitelist de views de infra que o LLM pode usar
ALLOWED_INFRA_VIEWS = {
    "DBA_DATA_FILES",
    "DBA_TEMP_FILES",
    "DBA_FREE_SPACE",
    "DBA_TABLESPACES",
    "DBA_SEGMENTS",
    "DBA_USERS",
    "DBA_DB_LINKS",
    "DBA_ROLE_PRIVS",
    "DBA_TAB_PRIVS",
    "DBA_HIST_ACTIVE_SESS_HISTORY",
    "DBA_HIST_SQLSTAT",

    "V$LOG",
    "V$CONTROLFILE",
    "V$INSTANCE",
    "V$DATABASE",
    "V$SESSION",
    "V$SYSTEM_EVENT",

    "GV$SESSION",
    "GV$SYSTEM_EVENT",
    "GV$ACTIVE_SESSION_HISTORY",
}


def is_infra_question(question: str) -> bool:
    """
    Classificação binária simples: é pergunta de INFRA?
    Não é determinístico, é só sugestão.
    """
    q = question.lower()

    keywords_infra = [
        "tablespace", "datafile", "datafiles", "segmento", "segmentos",
        "sessão", "sessao", "sessões", "sessoes", "session", "sessions",
        "espera", "wait", "eventos de espera", "system_event",
        "usuarios do banco", "usuários do banco", "dba_users",
        "dblink", "dblinks", "database link", "database links", "db links",
        "tamanho do banco", "tamanho do database",
        "redo log", "redolog", "controlfile", "control file",
        "uso de espaço", "space usage", "top objetos", "maiores tabelas",
        "maiores objetos", "auditoria", "audit", "ash", "awr",
    ]

    return any(k in q for k in keywords_infra)


def extract_tables_from_sql(sql: str) -> List[str]:
    """
    Extrai nomes de objetos nas cláusulas FROM e JOIN (simples).
    Considera apenas o primeiro token depois de FROM/JOIN.
    """
    upper = sql.upper()
    tables = []

    for m in re.finditer(r"\bFROM\s+([A-Z0-9_$#\"\.]+)", upper):
        tables.append(m.group(1))

    for m in re.finditer(r"\bJOIN\s+([A-Z0-9_$#\"\.]+)", upper):
        tables.append(m.group(1))

    return tables


def is_infra_sql_only(sql: str) -> bool:
    """
    Garante que todas as referências de FROM/JOIN são DBA_*, V$, GV$, CDB_*,
    ou estão na ALLOWED_INFRA_VIEWS.
    """
    tables = extract_tables_from_sql(sql)
    for t in tables:
        t_clean = t.strip('"')
        base = t_clean.split(".")[-1]

        if (
            base.startswith("DBA_")
            or base.startswith("V$")
            or base.startswith("GV$")
            or base.startswith("CDB_")
            or base.startswith("CDB$")
            or base in ALLOWED_INFRA_VIEWS
        ):
            continue

        print(f"[WARN] Objeto não permitido em SQL de infra: {t}")
        return False

    return True


def build_infra_prompt(question: str) -> str:
    """
    Prompt genérico para perguntas de infra.
    Única lógica para qualquer assunto de infra.
    """
    allowed_str = ", ".join(sorted(ALLOWED_INFRA_VIEWS))

    prompt = f"""
Você é um DBA Oracle sênior que gera SQL de infraestrutura para Oracle Database.

PERGUNTA DO USUÁRIO (sobre infra, administração, performance ou segurança):
\"\"\"{question}\"\"\"

REGRAS:
- Gere APENAS um comando SQL Oracle.
- O comando deve ser apenas SELECT ou WITH (sem INSERT/UPDATE/DELETE/MERGE, sem DDL).
- Você DEVE usar apenas visões de infraestrutura (sem tabelas de negócio).
- Prefira usar as seguintes views quando fizer sentido:
  {allowed_str}
- Use sempre sintaxe Oracle (por exemplo, FETCH FIRST N ROWS ONLY, funções Oracle etc.).
- Você pode usar funções Oracle (ROUND, SUM, COUNT, TRUNC, SYSDATE, etc.).

EXEMPLOS DE TIPOS DE CONSULTA (não copie literalmente, adapte à pergunta):

-- Tamanho total do banco (datafiles + tempfiles + redo logs + controlfiles)
WITH df AS (
  SELECT NVL(SUM(bytes), 0) AS bytes FROM dba_data_files
),
tf AS (
  SELECT NVL(SUM(bytes), 0) AS bytes FROM dba_temp_files
),
rl AS (
  SELECT NVL(SUM(bytes), 0) AS bytes FROM v$log
),
cf AS (
  SELECT NVL(SUM(block_size * file_size_blks), 0) AS bytes FROM v$controlfile
)
SELECT
  ROUND((df.bytes + tf.bytes + rl.bytes + cf.bytes) / (1024 * 1024 * 1024), 2) AS total_gb
FROM df, tf, rl, cf;

-- Usuários bloqueados no banco
SELECT username, account_status, lock_date, expiry_date
FROM dba_users
WHERE account_status LIKE 'LOCKED%';

-- DB links cadastrados
SELECT owner, db_link, username, host
FROM dba_db_links
ORDER BY owner, db_link;

-- Sessões de usuário por status
SELECT status, COUNT(*) AS qtd
FROM gv$session
WHERE type = 'USER'
GROUP BY status;

TAREFA:
- Com base na pergunta do usuário, gere UM comando SQL Oracle que responda à pergunta,
  usando apenas views de infraestrutura (DBA_*, V$, GV$, CDB_*).
- NÃO inclua ponto e vírgula no final.
- NÃO escreva explicações, apenas o SQL.
"""
    return prompt


def infra_nl_to_sql(question: str) -> Optional[str]:
    """
    Converte pergunta de infra em SQL via LLM, com validação forte.
    """
    prompt = build_infra_prompt(question)
    raw = call_llm(prompt)
    sql = strip_sql_from_llm(raw)

    if not is_safe_select(sql):
        print("[ERRO] SQL de infra não é SELECT/WITH. Bloqueado.")
        return None

    if not is_infra_sql_only(sql):
        print("[ERRO] SQL de infra referencia objetos fora de DBA_*/V$/GV$/CDB_. Bloqueado.")
        return None

    return sql


# ==========================
# MODO NEGÓCIO: NL2SQL usando metadados + LLM
# ==========================

def build_business_prompt(question: str, table_meta: TableMeta) -> str:
    cols_desc = []
    for col in table_meta.columns.values():
        cols_desc.append(
            f"- {col.name} ({col.data_type}"
            + (f"({col.data_length})" if col.data_length else "")
            + (f") - {col.comment}" if col.comment else ")")
        )

    prompt = f"""
Você é um assistente especializado em gerar planos de SELECT para Oracle Database em uma tabela de negócio.

Banco de dados alvo: Oracle Database.

Tabela alvo:
  - OWNER: {table_meta.owner}
  - TABLE: {table_meta.name}
  - Comentário da tabela: {table_meta.comment or "sem comentário"}

Colunas disponíveis:
{os.linesep.join(cols_desc)}

Usuário perguntou (em linguagem natural):
\"\"\"{question}\"\"\"


TAREFA:
- Escolha as colunas mais relevantes para o SELECT usando APENAS as colunas listadas.
- Se a pergunta indicar filtros (período, status, loja, cliente, etc.),
  monte as condições em SQL Oracle, SEM a palavra WHERE.
- Não invente nomes de colunas.
- Não invente outras tabelas ou JOINs.
- Pode usar funções Oracle básicas (TRUNC, SYSDATE, ADD_MONTHS, BETWEEN etc.).
- Se o filtro envolver valores dinâmicos (ex: código da loja informado pelo usuário),
  prefira BINDs (ex: :COD_LOJA).

RETORNO:
Responda APENAS com um JSON válido, no formato:

{{
  "select_columns": ["COLUNA1", "COLUNA2", "..."],
  "where_clauses": ["COLUNAX = 'VALOR'", "COLUNA_Y >= ADD_MONTHS(TRUNC(SYSDATE), -1)", "..."]
}}

- "select_columns": lista de colunas para o SELECT (nomes exatos).
- "where_clauses": lista de condições SQL (sem WHERE). Pode ser lista vazia.

NÃO escreva explicações, apenas o JSON.
"""
    return prompt


def parse_llm_json(raw: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def build_sql_from_business_plan(table_meta: TableMeta, plan: dict) -> Optional[str]:
    select_columns = plan.get("select_columns") or []
    where_clauses = plan.get("where_clauses") or []

    # valida colunas
    valid_cols = []
    for col in select_columns:
        col_up = str(col).upper()
        if col_up in table_meta.columns:
            valid_cols.append(col_up)
        else:
            print(f"[WARN] Coluna sugerida não existe na tabela e será ignorada: {col_up}")

    if not valid_cols:
        print("[INFO] Nenhuma coluna válida sugerida. Usando algumas colunas padrão.")
        valid_cols = sorted(list(table_meta.columns.keys()))[:5]

    select_list = ", ".join(valid_cols)
    fq_table = f"{table_meta.owner}.{table_meta.name}"

    all_col_names = set(table_meta.columns.keys())
    final_wheres = []
    for cond in where_clauses:
        cond_str_up = str(cond).upper()
        if any(c in cond_str_up for c in all_col_names):
            final_wheres.append(str(cond))
        else:
            print(f"[WARN] Condição ignorada (não parece referenciar colunas da tabela): {cond}")

    base_select = f"SELECT {select_list}\nFROM {fq_table}"

    if final_wheres:
        where_str = "\nWHERE " + "\n  AND ".join(final_wheres)
    else:
        where_str = ""

    return base_select + where_str


def business_nl_to_sql(question: str, metadata: MetadataCache, conn) -> Optional[str]:
    tokens = tokenize(question)
    best = metadata.find_best_table(tokens)

    if best is None:
        print("\n[INFO] Não encontrei tabela compatível no dicionário.")
        owner = input("Informe o SCHEMA (OWNER) da tabela (ex: APP): ").strip().upper()
        table = input("Informe o NOME da tabela (ex: CLIENTES): ").strip().upper()

        key = metadata.key(owner, table)
        if key not in metadata.tables:
            print(f"[INFO] Recarregando metadados do owner {owner}...")
            extra_meta = load_metadata(conn, owners_filter=[owner])
            for k, t in extra_meta.tables.items():
                metadata.tables[k] = t

        if key not in metadata.tables:
            print("[ERRO] Mesmo após recarregar, não tenho metadados dessa tabela.")
            return None

        table_meta = metadata.tables[key]
    else:
        table_key, table_meta, score = best
        print(f"\n[INFO] Tabela inferida pelo dicionário: {table_key} (score={score})")

    print("[INFO] Chamando LLM para montar SELECT (colunas + condições) na tabela de negócio...")
    prompt = build_business_prompt(question, table_meta)
    raw = call_llm(prompt)

    plan = parse_llm_json(raw)
    if plan is None:
        print("[ERRO] Não consegui interpretar o JSON retornado pelo LLM.")
        print("Resposta bruta:")
        print(raw)
        return None

    sql = build_sql_from_business_plan(table_meta, plan)
    return sql


# ==========================
# Escolha de modo (INFRA x NEGÓCIO)
# ==========================

def choose_mode(question: str) -> str:
    """
    Pergunta ao usuário se a consulta deve ser tratada como:
    - NEGÓCIO (tabelas da aplicação)
    - INFRA (DBA_*, V$, GV$)

    Usa is_infra_question() como sugestão, mas quem manda é o usuário.
    Retorna "infra" ou "negocio".
    """
    guessed_infra = is_infra_question(question)
    sugestao = "INFRA" if guessed_infra else "NEGÓCIO"

    print("\n[ASSISTENTE] Classificação sugerida pela engine:", sugestao)
    print("Como você quer tratar essa pergunta?")
    print("  [1] NEGÓCIO (tabelas da aplicação)")
    print("  [2] INFRA (DBA_*, V$, GV$)")

    choice = input(f"Opção [1/2] (ENTER = {sugestao}): ").strip()

    if choice == "":
        return "infra" if guessed_infra else "negocio"
    if choice == "2":
        return "infra"
    return "negocio"


def nl_to_sql(question: str, metadata: MetadataCache, conn) -> Optional[str]:
    """
    Fluxo principal:
    - Pergunta ao usuário se a consulta é INFRA ou NEGÓCIO
      (usando is_infra_question apenas como sugestão)
    - Se INFRA: usa infra_nl_to_sql (DBA_*, V$, GV$)
    - Se NEGÓCIO: usa business_nl_to_sql (tabelas de aplicação)
    """
    mode = choose_mode(question)

    if mode == "infra":
        print("\n[INFO] Modo selecionado: INFRA (DBA_*/V$/GV$/CDB_).")
        return infra_nl_to_sql(question)
    else:
        print("\n[INFO] Modo selecionado: NEGÓCIO (tabelas da aplicação).")
        return business_nl_to_sql(question, metadata, conn)


# ==========================
# Execução de SELECT (saídas diferentes)
# ==========================

def execute_select_rows(conn, sql: str, fetch_rows: int = 20):
    """
    Executa o SQL original e mostra as primeiras linhas.
    """
    if not is_safe_select(sql):
        print("[ERRO] SQL não é SELECT/WITH seguro. Execução bloqueada.")
        return

    with conn.cursor() as cur:
        try:
            cur.execute(sql)
        except oracledb.Error as e:
            print("[ERRO] Falha ao executar o SQL:")
            print(e)
            return

        col_names = [d[0] for d in cur.description]
        rows = cur.fetchmany(fetch_rows)

        print("\n[RESULTADO] Primeiras linhas:")
        print(" | ".join(col_names))
        print("-" * (len(" | ".join(col_names)) + 5))

        for r in rows:
            print(" | ".join(str(v) if v is not None else "NULL" for v in r))


def execute_select_count(conn, sql: str):
    """
    Executa um COUNT(*) em cima do SQL original, sem alterar o SQL-base.
    """
    if not is_safe_select(sql):
        print("[ERRO] SQL não é SELECT/WITH seguro. Execução bloqueada.")
        return

    count_sql = f"""
SELECT COUNT(*) AS total_rows
FROM (
{sql}
)
"""

    with conn.cursor() as cur:
        try:
            cur.execute(count_sql)
        except oracledb.Error as e:
            print("[ERRO] Falha ao executar o COUNT(*):")
            print(e)
            return

        row = cur.fetchone()
        total = row[0] if row else 0
        print(f"\n[RESULTADO] Total de linhas retornadas pelo SQL-base: {total}")


# ==========================
# Main
# ==========================

def main():
    print("=== NL2SQL Oracle (SELECT / WITH) - Negócio + Infra ===")
    print("Conectando no banco...")
    try:
        validate_provider_configuration()
    except RuntimeError as e:  # noqa: BLE001
        print(f"[ERRO] Configuração inválida: {e}")
        sys.exit(1)

    conn = get_connection()

    owners_filter_input = input(
        "Owners (schemas) para carregar metadados (separados por vírgula, ou vazio para todos exceto SYS/SYSTEM): "
    ).strip()

    owners_filter = None
    if owners_filter_input:
        owners_filter = [o.strip().upper() for o in owners_filter_input.split(",") if o.strip()]

    metadata = load_metadata(conn, owners_filter=owners_filter)

    print(f"Provider LLM: {get_llm_provider()}")

    while True:
        print("\n----------------------------------------")
        question = input("Pergunta em linguagem natural (ou 'sair'): ").strip()
        if question.lower() == "sair":
            break
        if not question:
            continue

        sql = nl_to_sql(question, metadata, conn)
        if not sql:
            print("[INFO] Não consegui gerar uma query segura.")
            continue

        print("\n[SQL SUGERIDO]:")
        print(sql)

        print("\nQual saída você quer ver desse SQL?")
        print("  [1] Primeiras linhas (default)")
        print("  [2] Apenas contagem de linhas (COUNT(*))")
        print("  [3] Primeiras linhas + COUNT(*)")

        choice = input("Opção [1/2/3] (ENTER = 1): ").strip()
        if choice == "":
            choice = "1"

        if choice == "2":
            execute_select_count(conn, sql)
        elif choice == "3":
            execute_select_rows(conn, sql)
            execute_select_count(conn, sql)
        else:
            execute_select_rows(conn, sql)

    conn.close()
    print("Conexão encerrada. Fim.")


if __name__ == "__main__":
    main()
