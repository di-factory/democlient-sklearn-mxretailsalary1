# RAW DATA METADATA
## name-of-file.csv: short desc
## raw-data.csv: Datos del censo economico del INEGI 2019, con actualizaciones desde 2008 (preprocessing)
## original fields:
* **field**: type, "desc (num)"
* **estado**: cat, "estado de Mexico (32)"
* **municipio**: cat, "municipio de cada estado en Mexico (3000)"
* **businesses**: int, "numero de negocios o unidades economicas"
* **employees**, int, "numero TOTAL de empleados en esas unidades economicas"
* **Payroll**, float, "monto en millones de pesos, del pago ANUAL de remuneraciones a los empleados"
* **expenditures**, float, "monto en millones, correspondientes al pago ANUAL de TODOS los gastos de los negocios"
* **income**, float, "monto en millones, es el ingreso ANUAL por ventas de los negocios"

## preprocessed fields:

* **payroll_day**, float, "monto del pago de las remuneraciones por empleado y dia"
* **profits_biz_day**, float, "monto de las ganancias (utilidad de operacion) de cada negocio por dia"
* **sales_employee_day**, float, "monto de las ventas generadas por cada empleado al negocio por dia"
* **employees_unit**, float, "numero de empleados por negocio"


## From the source (SOURCE: INEGI):
identifier: MEX-INEGI.CNE.03.05-CE-2019 title: Censos Económicos 2019. description:INEGI. Censos Económicos 2019. SNIEG. Información de Interés Nacional. Los Censos Económicos 2019 representan el decimonoveno evento censal y su objetivo consiste en obtener información estadística básica, referida al año 2018, sobre todos los establecimientos productores de bienes, comercializadores de mercancías y prestadores de servicios, para generar indicadores económicos de México a un gran nivel de detalle geográfico, sectorial y temático. Con el Acuerdo 3a/VIII/2009, del 25 de marzo de 2009, se considera como Información de Interés Nacional a partir de los Censos Económicos 2004, en virtud de que cumple con los cuatro criterios establecidos en el artículo 78 de la Ley del Sistema Nacional de Información Estadística y Geográfica.

Los Resultados Oportunos de los Censos Económicos 2019, presentan información de interés sobre las principales variables económicas a nivel nacional y entidad federativa, como el número de establecimientos y de su personal ocupado, el valor total de las remuneraciones, los gastos y los ingresos en millones de pesos. Además, se incluyen una gran variedad de productos con información de interés para el usuario como son los siguientes: Minimonografía de Resultados Nacionales; Infografías; Folleto de Resultados Oportunos; la publicación de Resultados Oportunos de los Censos Económicos 2019; y la presentación con información que contiene gráficas y mapas de manera clara y sencilla para ser entendido por los usuarios.

Se presentan 15 tabulados con información oportuna, lista para ser explotada por todo tipo de usuarios. Los primeros tres tabulados presenta información del Universo total de establecimientos por año de inicio de actividades, tipo de recorrido y tipo de propiedad, dividida en hombres y mujeres. El resto de tabulados presentan una amplia gama de información económica a nivel nacional y entidad federativa, información desagregada a nivel sector, subsector y rama de actividad económica, de las variables de unidades económicas, personal ocupado, remuneraciones, gastos e ingresos.

Cobertura geográfica: Nacional y Entidad federativa

Se presenta información de todas las actividades económicas del país, de acuerdo con la agregación de sectores contenidos en el Sistema de Clasificación Industrial para América del Norte (SCIAN 2018): Las Industrias Manufactureras comprenden las actividades del sector 31-33 del SCIAN 2018, incluso las maquiladoras de bienes. Las actividades comerciales incluyen los sectores 43 Comercio al por mayor y 46 Comercio al por menor del SCIAN 2018.Los Servicios privados no financieros comprenden todos los servicios del sector privado, clasificados en los siguientes sectores de servicios del SCIAN 2018: 51 Información en medios masivos; 53 Servicios inmobiliarios y de alquiler de bienes muebles e intangibles; 54 Servicios profesionales, científicos y técnicos; 55 Corporativos; 56 Servicios de apoyo a los negocios y manejo de residuos, y servicios de remediación; 61 Servicios educativos; 62 Servicios de salud y de asistencia social; 71 Servicios de esparcimiento culturales y deportivos, y otros servicios recreativos; 72 Servicios de alojamiento temporal y de preparación de alimentos y bebidas; 81 Otros servicios excepto actividades gubernamentales. Dentro del Resto de actividades se encuentran: Pesca y acuicultura, y Servicios relacionados con las actividades agropecuarias y forestales (1125,1141 y 115); 21 Minería; 22 Generación, transmisión, distribución y comercialización de energía eléctrica, suministro de agua y de gas natural por ductos al consumidor final; 23 Construcción; 48-49 Transportes, correos y almacenamiento, y 52 Servicios financieros y de seguros.

La unidad de observación principal es el Establecimiento: es la unidad económica que en una sola ubicación física, asentada en un lugar de manera permanente y delimitada por construcciones e instalaciones fijas, combina acciones y recursos bajo el control de una sola entidad propietaria o controladora, para realizar actividades de producción de bienes, compra-venta de mercancías o prestación de servicios, sea con fines de lucro o no.

Gastos por consumo de bienes y servicios: Es el valor de todos los bienes y servicios consumidos por la unidad económica para realizar sus operaciones, independientemente del periodo en que hayan sido comprados. Incluye: el valor de los bienes y servicios que recibió de otros establecimientos de la misma empresa, con o sin costo (valorados a precio de mercado) para su uso en las actividades de producción u operación del establecimiento. Excluye: los gastos fiscales, financieros y donaciones.

Ingresos por suministro de bienes y servicios: Es el monto que obtuvo el establecimiento por todas aquellas actividades de producción de bienes, comercialización de mercancías y prestación de servicios. Incluye: el valor de los bienes y servicios transferidos a otros establecimientos de la misma empresa, valorados a precio de venta, más todas las erogaciones o impuestos cobrados al comprador. Excluye: los ingresos financieros, subsidios, cuotas, aportaciones y venta de activos fijos. La valoración se realiza a precio de venta, menos todas las concesiones otorgadas a los clientes, tales como: descuentos, bonificaciones y devoluciones, así como fletes, seguros y almacenamiento de los productos suministrados cuando se cobren de manera independiente. Excluye: el Impuesto al Valor Agregado (IVA).

Personal ocupado: Comprende a todas las personas que trabajaron durante el periodo de referencia dependiendo contractualmente o no de la unidad económica, sujetas a su dirección y control.

Propietarios, familiares y otros trabajadores no remunerados: Comprende a los familiares que trabajan para el establecimiento, bajo su dirección y control sin una remuneración fija y periódica (esto es sin un acuerdo monetario por el trabajo realizado), cubriendo como mínimo una tercera parte de la jornada laboral. También se incluyen aquí a las personas que trabajan bajo la dirección y control de la unidad económica sin una remuneración regular, al encontrarse en forma de meritorios, servicios de capacitación o entrenamiento para efectuar una actividad económica, becarios por el sistema nacional de empleo y personal que labora para la unidad económica percibiendo exclusivamente propinas y a los trabajadores voluntarios. Incluye a los propietarios individuales y socios activos que trabajan en el establecimiento, excluyendo a los socios inactivos cuya actividad principal es fuera del establecimiento.

Personal ocupado remunerado: Comprende a todas las personas que trabajaron durante el periodo de referencia dependiendo contractualmente de la unidad económica, sujetas a su dirección y control, a cambio de una remuneración fija y periódica por su participación en las actividades de producción, comercialización de mercancías, prestación de servicios y administración del establecimiento.

Remuneraciones: Son todos los pagos y aportaciones normales y extraordinarias, en dinero y especie, antes de cualquier deducción, para retribuir el trabajo del personal dependiente de la razón social, en forma de salarios y sueldos, prestaciones sociales y utilidades repartidas al personal, ya sea que este pago se calcule sobre la base de una jornada de trabajo o por la cantidad de trabajo desarrollado (destajo); o mediante un salario base que se complementa con comisiones por ventas u otras actividades. Incluye: las contribuciones patronales a regímenes de seguridad social; el pago realizado al personal con licencia y permiso temporal. Excluye: los pagos por liquidaciones o indemnizaciones, pagos a terceros por el suministro de personal ocupado; pagos exclusivamente de comisiones para aquel personal que no recibió un sueldo fijo; pagos de honorarios por servicios profesionales contratados de manera infrecuente.

Unidades económicas: Son las unidades estadísticas sobre las cuales se recopilan datos; se dedican principalmente a un tipo de actividad de manera permanente en construcciones e instalaciones fijas, combinando acciones y recursos bajo el control de una sola entidad propietaria o controladora, para llevar a cabo producción de bienes y servicios, sea con fines mercantiles o no. Se definen por sector de acuerdo con la disponibilidad de registros contables y la necesidad de obtener información con el mayor nivel de precisión analítica.

Para obtener los totales de establecimientos y personal ocupado se pueden calcular de acuerdo a:

RONACE19_01:

Total censal de establecimientos = Iniciaron en 2019 + Iniciaron antes del 2019
Total de personal ocupado que iniciaron en 2019 (B) = Personal ocupado hombres (C) + Personal ocupado mujeres (D)
Total de personal ocupado que iniciaron antes del 2019 (E) = Personal ocupado hombres (F) + Personal ocupado mujeres (G)
Total censal de personal ocupado (A) = Total (B) + Total (E); o bien sumando Personal ocupado hombres (C) + Personal ocupado mujeres (D) + Personal ocupado hombres (F) + Personal ocupado mujeres (G)
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
NOTA:La suma de los parciales puede no coincidir con el total, por una diferencia mínima; consecuencia del redondeo en los valores que contienen decimales, producto de la expansión de la muestra rural.

RONACE19_02:

Total de personal ocupado (A) = Personal ocupado hombres (B) + Personal ocupado mujeres (C)
Para calcular el total nacional, se suma el total de zona urbana y zona rural para cada variable.

RONACE19_03:

Total de personal ocupado (A) = Personal ocupado hombres (B) + Personal ocupado mujeres (C)
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de los 3 tipos de propiedad (Privado-paraestatal, Servicios públicos y Asociaciones religiosas) para cada variable.
Para obtener los totales de unidades económicas y personal ocupado se pueden calcular de acuerdo a:

RONACE19_04; RONACE19_05; RONACE19_06

Total de personal ocupado (A) = Total de personal ocupado dependiente de la razón social (B) + Total de personal ocupado no dependiente de la razón social (E); o bien sumando Personal ocupado remunerado (C) + Propietarios, familiares y otros trabajadores no remunerados (D) + Total de personal ocupado no dependiente de la razón social (E)
Total de personal ocupado dependiente de la razón social (B) = Personal ocupado remunerado (C) + Propietarios, familiares y otros trabajadores no remunerados (D)
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de los 12 estratos de personal ocupado para cada variable.

RONACE19_07:

Para calcular el total nacional, se suman el total de los sectores de manufacturas, comercio, servicios privados no financieros y resto de actividades económicas.
Para calcular el total de cada una de las actividades económicas se suma el total de los 4 tamaños de la unidad económica para cada variable.

RONACE19_08:

Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades se suma el total de los 4 tamaños de la unidad económica para cada variable.
La información del tabulado No considera Unidades pesqueras ni acuícolas.

RONACE19_9:

Para calcular el total nacional, se suman el total de los sectores de manufacturas, comercio, servicios privados no financieros y resto de actividades económicas.
Para calcular el total de cada una de las actividades económicas se suma el total de los 4 tamaños de la unidad económica para cada variable.

RONACE19_10:

Total de unidades económicas = Total de unidades económicas que realizaron compras o ventas por internet + Total de unidades económicas que no realizaron compras o ventas por internet
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.

RONACE19_11:

Total de unidades económicas = Total de unidades económicas que utilizaron un sistema contable + Total de unidades económicas que no utilizaron un sistema contable
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
La información del tabulado No considera Actividad de Comercio ni Manufacturas de Gobierno. Tampoco incluye Actividad económica en vivienda.

RONACE19_12:

Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
La información del tabulado No considera Actividad de Comercio ni Manufacturas de Gobierno. Tampoco incluye Actividad económica en vivienda.
Las unidades económicas pudieron reportar más de un problema.

RONACE19_13:

Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
La información del tabulado No considera Servicios financieros y de seguros de establecimientos grandes. Tampoco incluye Actividad de Comercio ni Manufacturas de Gobierno.
Las unidades económicas pudieron reportar más de una opción de respuesta.

RONACE19_14:

Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
La información del tabulado No considera Actividad de Comercio ni Manufacturas de Gobierno. Tampoco incluye Actividad económica en vivienda.

RONACE19_15:

Total de unidades económicas = Total de unidades económicas que capacitaron al personal ocupado + Total de unidades económicas que no capacitaron al personal ocupado
Para calcular el total nacional, se suma el total de las 32 entidades federativas para cada variable.
Para calcular el total de cada una de las entidades, se suma el total de las 4 actividades económicas (manufacturas, comercio, servicios privados no financieros y resto de actividades económicas) para cada variable.
La información del tabulado No considera Actividad de Comercio ni Manufacturas de Gobierno. Tampoco incluye Actividad económica en vivienda. modified: 10/12/2019 12:00:00 a. m. publisher: Instituto Nacional de Geografía y Estadística, INEGI mbox: atencion.usuarios@inegi.org.mx license: https://www.inegi.org.mx/inegi/terminos.html dataDictionary: keyword:censos, económicos, personal ocupado, remuneraciones, producción, gastos, ingresos, activos fijos, existencias, financiamiento, cuentas bancarias, industria, comercio, servicios, pesca, minería, construcción, electricidad, petróleo, agua, transportes, sector, subsector, rama de actividad, clase de actividad contactPoint: Lic. José de Jesús Esquivel de la Rosa, 800 111 4634 temporal: 2018 spatial: Estados Unidos Mexicanos actualPeriodicity: Quinquenal Distribution: https://inegi.org.mx/contenidos/masiva/indicadores/programas/ce/2019/