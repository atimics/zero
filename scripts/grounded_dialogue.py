"""Compile explicit source fields into four-turn dialogue training rows."""
import copy
from string import Formatter


def render(template, fields):
    """Bind numbered fields and retain exact UTF-8 copy spans."""
    known={f['field']:f for f in fields}
    text='';copies=[]
    for literal,slot,spec,conversion in Formatter().parse(template):
        text+=literal
        if slot is None:continue
        if not slot.isdigit() or spec or conversion:raise ValueError('Only numbered plain fields are supported')
        field=known[int(slot)]
        if not field.get('spoken') or field['role'] in (0,8):raise ValueError('Template asks for an unavailable spoken field')
        start=len(text.encode());text+=field['text']
        if field.get('knowledge',0)!=3:copies.append({**field,'start':start,'end':len(text.encode())})
    return text,copies


def answer_template(template, row):
    if row.get('confidence',80)<40:return template.rstrip('.')+', I think.'
    if row.get('retold'):return template.rstrip('.')+', from what I hear.'
    return template


def exchange_rows(source, templates, metadata):
    """Every target field mention comes from a template placeholder."""
    base=copy.deepcopy(source);base['kind_id']=metadata['meaning_ids'][base['rule']]
    result=[];lines=[]
    for turn,template in enumerate([None,*templates]):
        row=copy.deepcopy(base)
        if template is not None:
            if turn==2:template=answer_template(template,base)
            row['output'],row['copies']=render(template,base['fields'])
        row['history']=[{'speaker':'self' if j%2==turn%2 else 'other','text':line} for j,line in enumerate(lines)][-4:]
        row['id']=base['id']+':turn:'+str(turn)
        result.append(row);lines.append(row['output'])
    return result


def build(rows, bank, metadata):
    expected=set(metadata['meaning_ids'])
    if set(bank)!=expected or {r['rule'] for r in rows}!=expected:raise ValueError('Dialogue and source coverage must match the model grammar')
    if any(len(v)!=3 for v in bank.values()):raise ValueError('Each exchange needs question, answer, and reaction')
    return [turn for row in rows for turn in exchange_rows(row,bank[row['rule']],metadata)]


def replace_fields(row, replacements):
    """Change full fields in a diagnostic account, with exact byte offsets."""
    result=copy.deepcopy(row)
    def rewrite(text,spans):
        raw=text.encode();out=b'';updated=[];at=0
        for span in sorted(spans,key=lambda f:f['start']):
            if span['start']<at or raw[span['start']:span['end']].decode()!=span['text']:raise ValueError('Invalid source span')
            out+=raw[at:span['start']];start=len(out)
            value=replacements.get(span['field'],span['text']);out+=value.encode()
            updated.append({**span,'text':value,'start':start,'end':len(out)});at=span['end']
        return (out+raw[at:]).decode(),updated
    result['prefix'],result['fields']=rewrite(row['prefix'],row['fields'])
    result['output'],result['copies']=rewrite(row['output'],row['copies'])
    result['history']=[]
    return result
