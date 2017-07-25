---
layout: page
title: Contact
permalink: /contact/
locations:
 - Pevensey 3, Falmer, United Kingdom
zoom: 16
---

 
You can find me in office 4C14 in the Pevensey 3 building, on the University of Sussex campus.

## Telephone
+44 (0)1273 873087 

## Address
Department of Physics and Astronomy \\
University of Sussex \\
Brighton, \\
Sussex, \\
BN1 9QH \\
UK

{% if page.locations %} 
<img src="http://maps.googleapis.com/maps/api/staticmap?{% for location in page.locations %}{% if forloop.first %}center={{location}}&markers=color:blue%7C{{location}}{% else %}&markers=color:blue%7C{{location}}{% endif %}{% endfor %}&zoom={% if page.zoom %}{{page.zoom}}{% else %}13{% endif %}&size=500x500&scale=1.5&sensor=false&visual_refresh=true" alt="">
{% endif %}