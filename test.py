import spacy
import pytextrank

# example text
text = "The U.S. Food and Drug Administration (FDA) regulations establish limits for contaminants in bottled water that must provide the same protection for public health.[9] Drinking water, including bottled water, may reasonably be expected to contain at least small amounts of some contaminants. The presence of these contaminants does not necessarily indicate that the water poses a health risk. In urbanized areas around the world, water purification technology is used in municipal water systems to remove contaminants from the source water (surface water or groundwater) before it is distributed to homes, businesses, schools and other recipients. Water drawn directly from a stream, lake, or aquifer and that has no treatment will be of uncertain quality.Dissolved minerals may affect suitability of water for a range of industrial and domestic purposes. The most familiar of these is probably the presence of ions of calcium (Ca2+) and magnesium (Mg2+) which interfere with the cleaning action of soap, and can form hard sulfate and soft carbonate deposits in water heaters or boilers.[10]  Hard water may be softened to remove these ions. The softening process often substitutes sodium cations.[11]  Hard water may be preferable to soft water for human consumption, since health problems have been associated with excess sodium and with calcium and magnesium deficiencies. Softening decreases nutrition and may increase cleaning effectiveness.[12] Various industries' wastes and effluents can also pollute the water quality in receiving bodies of water.[13]"

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
tr = pytextrank.TextRank()
nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)

doc = nlp(text)

# examine the top-ranked phrases in the document
for p in doc._.phrases:
    print("{:.4f} {:5d}  {}".format(p.rank, p.count, p.text))
    print(p.chunks)
