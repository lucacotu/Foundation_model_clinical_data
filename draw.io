Una nota onesta sulla differenza tra i due strumenti: in draw.io non c'è un parametro "node distance" come in TikZ; la compattezza si ottiene avvicinando le coordinate x dei blocchi. Ho quindi ridotto le distanze orizzontali rispetto alla versione draw.io precedente. Inoltre, dato che hai sostituito i primi due blocchi con le tue immagini (table.png e deep-learning.png), qui uso due segnaposto-immagine: draw.io non incorpora il file PNG nell'XML, quindi dovrai trascinare tu le due immagini dentro quei due riquadri (o sostituire l'image=... con il percorso/URL), come ti spiego sotto.
xml<mxfile host="app.diagrams.net">
  <diagram name="Framework" id="framework-pipeline">
    <mxGraphModel dx="900" dy="600" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="460" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- ===== Tabular data (segnaposto immagine) ===== -->
        <mxCell id="data" value="" style="rounded=0;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#999999;verticalLabelPosition=bottom;verticalAlign=top;" vertex="1" parent="1">
          <mxGeometry x="40" y="120" width="90" height="70" as="geometry" /></mxCell>
        <mxCell id="datalbl" value="Tabular data" style="text;html=1;align=center;verticalAlign=top;" vertex="1" parent="1">
          <mxGeometry x="40" y="192" width="90" height="20" as="geometry" /></mxCell>

        <!-- ===== Foundation model (segnaposto immagine) ===== -->
        <mxCell id="fm" value="" style="rounded=0;whiteSpace=wrap;html=1;dashed=1;fillColor=none;strokeColor=#999999;verticalLabelPosition=bottom;verticalAlign=top;" vertex="1" parent="1">
          <mxGeometry x="190" y="120" width="90" height="70" as="geometry" /></mxCell>
        <mxCell id="fmlbl" value="Foundation model" style="text;html=1;align=center;verticalAlign=top;" vertex="1" parent="1">
          <mxGeometry x="180" y="192" width="110" height="20" as="geometry" /></mxCell>

        <!-- ===== Resto pipeline ===== -->
        <mxCell id="emb" value="Embeddings" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#eef3fc;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="340" y="123" width="95" height="60" as="geometry" /></mxCell>
        <mxCell id="head" value="Survival&#10;heads" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5e8d4;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="495" y="123" width="95" height="60" as="geometry" /></mxCell>
        <mxCell id="res" value="Risk estimate&#10;(C-index)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#d5f5d0;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="650" y="123" width="95" height="60" as="geometry" /></mxCell>

        <!-- ===== Ramo baseline ===== -->
        <mxCell id="base" value="Baseline&#10;survival models&#10;(raw data, no embeddings)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#ffe6cc;strokeColor=#000000;" vertex="1" parent="1">
          <mxGeometry x="495" y="300" width="95" height="70" as="geometry" /></mxCell>

        <!-- ===== Frecce percorso principale ===== -->
        <mxCell id="e1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="data" target="fm"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0;entryY=0.5;" edge="1" parent="1" source="fm" target="emb"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e3" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;" edge="1" parent="1" source="emb" target="head"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e4" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;" edge="1" parent="1" source="head" target="res"><mxGeometry relative="1" as="geometry" /></mxCell>

        <!-- ===== Frecce ramo baseline (parte da sotto l'etichetta) ===== -->
        <mxCell id="e5" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;exitX=0.5;exitY=1;entryX=0;entryY=0.5;" edge="1" parent="1" source="datalbl" target="base"><mxGeometry relative="1" as="geometry" /></mxCell>
        <mxCell id="e6" style="edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;strokeWidth=1.5;exitX=1;exitY=0.5;entryX=0.5;entryY=1;" edge="1" parent="1" source="base" target="res"><mxGeometry relative="1" as="geometry" /></mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
Come usarlo e cosa ho cambiato.
Le distanze orizzontali sono state ridotte: i blocchi ora distano circa 60px l'uno dall'altro (prima erano ~80px), così il diagramma è più compatto in larghezza, in linea con la compressione che hai fatto in TikZ. La freccia verso le baseline (e5) ora parte dal nodo datalbl, cioè dall'etichetta "Tabular data", uscendo dal suo lato inferiore (exitX=0.5;exitY=1): così nasce sotto la scritta e non la attraversa, esattamente come nella versione TikZ.
Per le due immagini: i blocchi data e fm sono ora due riquadri tratteggiati vuoti che fanno da segnaposto. Per inserire le tue PNG hai due modi. Il più semplice: in draw.io trascini table.png e deep-learning.png dentro i rispettivi riquadri, oppure selezioni il riquadro e usi Edit Style sostituendo lo stile con shape=image;image=<percorso-o-URL>;. In alternativa, se le immagini sono raggiungibili via URL, puoi mettere direttamente image=https://.../table.png nello stile. Ho lasciato i segnaposto tratteggiati con l'etichetta sotto, così la struttura e le frecce sono già pronte e tu devi solo agganciare le immagini.
Una differenza che noterai rispetto a TikZ: in TikZ le immagini erano già incluse via \includegraphics, qui invece il file PNG non vive dentro l'XML, quindi il passaggio manuale di trascinare le immagini è inevitabile. È il motivo per cui, per la tesi, la via TikZ resta più comoda (le immagini sono referenziate dal sorgente e si aggiornano da sole); questa versione draw.io è utile se vuoi ritoccare il layout a mano prima di esportare.