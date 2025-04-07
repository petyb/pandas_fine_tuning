import ast
import json


def get_functions(path):
    with open('pandas/' + path, 'r', encoding='utf-8') as f:
        lines = f.read()
    tree = ast.parse(lines)
    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node):
            start = node.lineno
            end = max(getattr(i, 'lineno', start) for i in ast.walk(node))
            function = "\n".join(lines.splitlines()[start - 1:end])
            docstring = ast.get_docstring(node)
            result.append({
                "docstring": docstring,
                "function_name": node.name,
                "code": function,
                "start_line": start,
                "end_line": end,
                "file_path": str(path)
            })
    return result


data = []
with open('filelist.txt', 'r', encoding='utf-8') as f:
    for line in f.read().split('\n'):
        data += get_functions(line)

with open("data/data.jsonl", "w") as f:
    for sample in data:
        f.write(json.dumps(sample) + "\n")
