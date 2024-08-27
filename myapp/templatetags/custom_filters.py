from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    return dictionary.get(key)

@register.simple_tag
def get_dynamic_field(form, base_name, index):
    field_name = f'{base_name}_{index}'
    return form[field_name]

@register.filter
def get_field(form, field_name):
    return form[field_name]