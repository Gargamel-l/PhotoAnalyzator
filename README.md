# PhotoAnalyzator
алгоритм рабоыт программы:

1.Предобработка изображений: Преобразование всех изображений к единому размеру и, возможно, к градациям серого для уменьшения размерности данных.

2.Извлечение признаков: Использование предварительно обученной модели глубокого обучения (например, ResNet, VGG16, или MobileNet) для извлечения признаков из изображений. Эти признаки представляют собой векторы, которые кодируют содержимое изображений.

3.Сравнение изображений: Сравнение векторов признаков изображений, загруженных пользователем, с векторами признаков изображений в БД, используя метрику сходства (например, Евклидово расстояние или косинусное сходство).

4.Определение сходства: Установление порога сходства, при котором изображения считаются схожими, и отбор изображений, которые удовлетворяют этому критерию.
