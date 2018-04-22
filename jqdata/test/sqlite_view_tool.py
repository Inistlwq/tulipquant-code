from sqlalchemy import *
from sqlalchemy.schema import DDLElement
from sqlalchemy.sql import table
from sqlalchemy.ext import compiler

class CreateView(DDLElement):
    def __init__(self, name, selectable):
        self.name = name
        self.selectable = selectable

class DropView(DDLElement):
    def __init__(self, name):
        self.name = name

@compiler.compiles(CreateView)
def compile(element, compiler, **kw):
    return "CREATE VIEW %s AS %s" % (element.name, compiler.sql_compiler.process(element.selectable))

@compiler.compiles(DropView)
def compile(element, compiler, **kw):
    return "DROP VIEW %s" % (element.name)

def view(name, metadata, selectable):
    t = table(name)

    for c in selectable.c:
        c._make_proxy(t)

    CreateView(name, selectable).execute_at('after-create', metadata)
    DropView(name).execute_at('before-drop', metadata)
    return t

if __name__ == '__main__':

    engine = create_engine("sqlite:////tmp/testdb.db", echo=True)
    metadata = MetaData(engine)
    stuff = Table('stuff', metadata,
        Column('id', Integer, primary_key=True),
        Column('data', String(50)),
    )

    more_stuff = Table('more_stuff', metadata,
        Column('id', Integer, primary_key=True),
        Column('stuff_id', Integer, ForeignKey('stuff.id')),
        Column('data', String(50)),
    )
    # SQLite requires you to explicitly give a label to a column in a select list otherwise the column name
    # is undefined and this breaks this demo as the query cannot find the column 'stuff_view.data'
    # See http://www.sqlite.org/c3ref/column_name.html
    stuff_view = view("stuff_view", metadata,
                    select([stuff.c.id.label('id'), stuff.c.data.label('data'), more_stuff.c.data.label('moredata')]).\
                    select_from(stuff.join(more_stuff)).\
                    where(stuff.c.data.like(text('"%orange%"'))
                    )
                )

    # the ORM would appreciate this
    assert stuff_view.primary_key == [stuff_view.c.id]

    metadata.create_all()

    stuff.insert().execute(
        {'data':'apples'},
        {'data':'pears'},
        {'data':'oranges'},
        {'data':'orange julius'},
        {'data':'apple jacks'},
    )

    more_stuff.insert().execute(
        {'stuff_id':3, 'data':'foobar'},
        {'stuff_id':4, 'data':'foobar'}
    )

    assert set(
            r[0:2] for r in engine.execute(select([stuff_view.c.data, stuff_view.c.moredata])).fetchall()
        ) == set([('oranges', 'foobar'), ('orange julius', 'foobar')])

    # illustrate ORM usage
    from sqlalchemy.orm import Session
    from sqlalchemy.ext.declarative import declarative_base
    Base = declarative_base(metadata=metadata)

    class MyStuff(Base):
        __table__ = stuff_view

    s = Session(engine)
    print s.query(MyStuff).all()

    # metadata.drop_all()
