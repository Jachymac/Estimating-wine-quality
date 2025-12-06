# Estimating-wine-quality
## Možné cíle projektu
- Aplikace vhodných způsobů předzpracování dat pro jejich následnou validní analýzu
- Výběr příznaků souvisejících s kvalitou vzorků vín
- Stanovení souvislostí mezi různými vlastnostmi vzorků
- Identifikace a interpretace různých skupin vzorků (nebo vlastností vzorků) na základě dostupných údajů
- Doporučení pro dosažení nejlepší kvality vína
- Efektivní vizualizace naměřených multimodálních dat

## Datový soubor
**red_wine_quality_Final.csv** – údaje o vlastnostech vzorků červeného vína a výsledného hodnocení
kvality vína na škále 0 až 10:
1. *fixedAcidity*:  netěkavé kyseliny
2. *volatileAcidity*:  těkavé kyseliny
3. *citricAcid*:  kyselina citrónová
4. *residualSugar*:  redukující cukry
5. *chlorides*:  chloridy
6. *freeSulfurDioxide*:  volný obsah oxidu siřičitého
7. *totalSulfurDioxide*:  celkový obsah oxidu siřičitého
8. *density*:  hustota
9. *pH*:  hodnota pH
10. *sulphates*:  sulfáty
11. *alcohol*:  obsah alkoholu
12. *quality*:  skóre (0 až 10)

### **POZOR**
V datovém souboru se mohou vyskytovat odlehlé hodnoty, chybějící hodnoty měřených
veličin (označené jako např. NaN, ‘-’ apod.), hodnoty nedávající smysl apod.
