
def split_facts(facts):
    heads = facts[:, 0]
    rels = facts[:, 1]
    tails = facts[:, 2]
    years = facts[:, 3]
    months = facts[:, 4]
    days = facts[:, 5]

    return heads, rels, tails, years, months, days
