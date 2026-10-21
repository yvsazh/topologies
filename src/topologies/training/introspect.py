"""Introspection APIs powering CLI help, docs tables, and future UIs."""

from __future__ import annotations


def list_recipes() -> list[dict[str, str]]:
    from topologies.training.recipe_base import list_recipes as _list

    return _list()


def recipe_schema(name: str) -> dict:
    from topologies.training.recipe_base import recipe_schema as _schema

    return _schema(name)


def all_recipe_schemas() -> dict[str, dict]:
    return {r["name"]: recipe_schema(r["name"]) for r in list_recipes()}
