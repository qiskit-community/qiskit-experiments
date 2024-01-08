{# This is identical to class.rst, except for the filtering of the inherited_members. -#}

{% if referencefile %}
.. include:: {{ referencefile }}
{% endif %}

{{ objname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :no-members:
   :no-inherited-members:
   :no-special-members:

{% block attributes_summary %}

   {% set wanted_attributes = [] %}
   {% for item in attributes%}
      {%- if not item.startswith('_') %}
         {% set _ = wanted_attributes.append(item)%}
      {%- endif -%}
   {%- endfor %}

   {% if wanted_attributes%}
   .. rubric:: Attributes
      {% for item in wanted_attributes %}
   .. autoattribute:: {{ name }}.{{ item }}
      {%- endfor %}
   {% endif %}
{% endblock %}

{% block methods_summary %}

   {% set wanted_methods = [] %}
   {% for item in all_methods %}
      {%- if item not in inherited_members %}
         {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__'] %}
            {% set _ = wanted_methods.append(item)%}
         {%- endif -%}
      {%- endif -%}
   {%- endfor %}

   {% if wanted_methods %}
   .. rubric:: Methods
   {% for item in wanted_methods %}
   .. automethod:: {{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
{% endblock %}