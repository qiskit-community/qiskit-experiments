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
   {% if attributes %}

   {% set count = namespace(value=0) %}

   {% for item in attributes %}
      {% if not item.startswith('_') %}
      {% set count.value = count.value + 1 %}
         {% if count.value == 1 %}
   .. rubric:: Attributes

   .. autosummary::
      :toctree: ../stubs/
         {% endif %}
      
      {{ name }}.{{ item }}
      {% endif %}
   {% endfor %}
   {% endif %}
   {% endblock %}

   {% block methods_summary %}
   {% if methods %}

   .. rubric:: Methods

   .. autosummary::
      :toctree: ../stubs/
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__', '__mul__', '__getitem__', '__len__'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% for item in inherited_members %}
      {%- if item in ['__call__', '__mul__', '__getitem__', '__len__'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}
