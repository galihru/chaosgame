import java.awt.{AlphaComposite, BasicStroke, Color, Dimension, Font, Graphics, Graphics2D, RenderingHints}
import java.awt.event.{ActionEvent, ActionListener, KeyEvent, KeyListener, MouseAdapter, MouseEvent, MouseWheelEvent, MouseWheelListener}
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import javax.swing.{BorderFactory, BoxLayout, JButton, JCheckBox, JComponent, JDialog, JFrame, JLabel, JPanel, JRadioButton, JSlider, SwingUtilities, Timer, WindowConstants, ButtonGroup}
import javax.swing.event.ChangeListener
import scala.collection.mutable
import scala.util.Random
import scala.math.{max, min, log, sqrt}

// 主程序
object AdvancedChaosGame {
  final val WIDTH = 1100
  final val HEIGHT = 800
  final val DEFAULT_VERTICES = 3
  final val DEFAULT_RATIO = 0.5
  final val DEFAULT_ITERS_PER_TICK = 200

  // 模式枚举
  enum Mode { case CLASSIC, AI_SOFTMAX, AI_UCB1, HYBRID }

  // 顶点类
  class Vertex(var x:Double, var y:Double, val color:Color)

  // Softmax Bandit（简明实现）
  class SoftmaxBandit(var k:Int) {
    var weights = Array.fill(k)(1.0)
    val rand = new Random()
    def sample(temperature:Double): Int = {
      val maxw = weights.max
      val exps = weights.map(w => math.exp((w - maxw)/max(1e-12, temperature)))
      val z = exps.sum
      var r = rand.nextDouble()*z
      var i = 0
      while (i < exps.length) { if (r < exps(i)) return i; r -= exps(i); i += 1 }
      exps.length - 1
    }
    // 更新规则（命名为 w_i <- w_i + α (reward - 0.5)）
    def update(index:Int, reward:Double, lr:Double): Unit = if (index >= 0 && index < weights.length) weights(index) += lr*(reward - 0.5)
    def normalize(): Unit = { val s = weights.sum; if (s > 0) for (i <- weights.indices) weights(i) = weights(i) / s * weights.length }
    def resize(newK:Int): Unit = {
      val w = Array.fill(newK)(1.0)
      val copyLen = math.min(weights.length, newK)
      Array.copy(weights, 0, w, 0, copyLen)
      weights = w; k = newK
    }
  }

  // UCB1 Bandit（简明实现）
  class UCB1Bandit(var k:Int) {
    var counts = Array.fill(k)(1L)
    var values = Array.fill(k)(0.5)
    var total = k
    val rand = new Random()
    // 选择：score = μ_i + sqrt(2 ln N / n_i)
    def sample(): Int = {
      var best = 0; var bestScore = Double.NegativeInfinity
      var i = 0
      while (i < counts.length) {
        val avg = values(i); val c = counts(i)
        val score = avg + math.sqrt(2.0 * math.log(max(1.0, total.toDouble)) / c)
        if (score > bestScore) { bestScore = score; best = i }
        i += 1
      }
      best
    }
    def update(index:Int, reward:Double): Unit = if (index >= 0 && index < counts.length) { values(index) = (values(index)*counts(index) + reward) / (counts(index)+1); counts(index) += 1; total += 1 }
    def resize(newK:Int): Unit = {
      val nc = Array.fill(newK)(1L); val nv = Array.fill(newK)(0.5)
      val copyLen = math.min(counts.length, newK)
      Array.copy(counts, 0, nc, 0, copyLen); Array.copy(values, 0, nv, 0, copyLen)
      counts = nc; values = nv; k = newK
    }
  }

  // 主画布面板（包含中文 tooltip）
  class ChaosPanel extends JComponent with ActionListener with KeyListener {
    setPreferredSize(new Dimension(WIDTH, HEIGHT))
    var persistentBuffer = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB)
    var transientBuffer = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB)
    var persistentG = persistentBuffer.createGraphics(); persistentG.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    var transientG = transientBuffer.createGraphics(); transientG.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    val bgColor = new Color(6,8,10)
    val latestDotColor = new Color(255,220,40,255)
    clearBuffers()

    // 状态与参数（中文）
    var mode:Mode = Mode.HYBRID
    var nVertices = DEFAULT_VERTICES
    var ratio = DEFAULT_RATIO
    var itersPerTick = DEFAULT_ITERS_PER_TICK
    var running = true
    var forbidSame = true
    var forbidNeighbors = false
    var useMidpointMode = false
    var showLatestDot = true
    var trailFade = true
    var colorMode = 0
    val rand = new Random()
    var vertices = mutable.ArrayBuffer.empty[Vertex]
    var curX = WIDTH/2.0; var curY = HEIGHT/2.0
    var lastIndex = -1
    var softmax = new SoftmaxBandit(nVertices)
    var ucb = new UCB1Bandit(nVertices)
    var temperature = 0.8
    var learningRate = 0.25
    var ratioPerVertex = Array.fill(10)(ratio)
    var totalPoints = 0L
    var rewardSum = 0.0
    var stepsSinceReset = 0
    val overlayComposite = AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.06f)
    var maxFillAttempts = 12
    var autoRamp = true
    var rampTarget = 15000
    var rampSpeed = 1.05
    var frames = 0

    // HYBRID epsilon（使用中文 tooltip 控制）
    var epsilon = 0.04
    var rewardRadius = 4

    // 视图变换（缩放/平移）
    var scale = 1.0
    var offsetX = 0.0
    var offsetY = 0.0
    var dragging = false
    var lastDragX = 0; var lastDragY = 0
    var showHeatmap = false
    var drawPointSizeWhenZoomed = true

    val timer = new Timer(16, this)
    addKeyListener(this); setFocusable(true)

    // 鼠标控制：左拖平移，滚轮缩放，右键设置起始点
    addMouseListener(new MouseAdapter {
      override def mousePressed(e: MouseEvent): Unit = {
        if (e.getButton == MouseEvent.BUTTON1) { dragging = true; lastDragX = e.getX; lastDragY = e.getY }
        if (e.getButton == MouseEvent.BUTTON3) {
          val lx = ((e.getX - offsetX) / scale).toInt; val ly = ((e.getY - offsetY) / scale).toInt
          if (lx >= 0 && lx < WIDTH && ly >= 0 && ly < HEIGHT) { curX = lx; curY = ly }
        }
      }
      override def mouseReleased(e: MouseEvent): Unit = { dragging = false }
    })
    addMouseMotionListener(new java.awt.event.MouseMotionAdapter {
      override def mouseDragged(e: java.awt.event.MouseEvent): Unit = {
        if (dragging) { val dx = e.getX - lastDragX; val dy = e.getY - lastDragY; offsetX += dx; offsetY += dy; lastDragX = e.getX; lastDragY = e.getY; repaint() }
      }
    })
    addMouseWheelListener(new MouseWheelListener { def mouseWheelMoved(e: MouseWheelEvent): Unit = {
      val notches = e.getWheelRotation
      val oldScale = scale
      if (notches < 0) scale = min(12.0, scale * math.pow(1.08, -notches)) else scale = max(0.05, scale * math.pow(0.92, notches))
      val mx = e.getX; val my = e.getY
      val worldX = (mx - offsetX) / oldScale; val worldY = (my - offsetY) / oldScale
      offsetX = mx - worldX * scale; offsetY = my - worldY * scale
      repaint()
    } })

    initVertices(); timer.start()

    def clearBuffers(): Unit = {
      val g1 = persistentBuffer.createGraphics(); g1.setColor(bgColor); g1.fillRect(0,0,WIDTH,HEIGHT); g1.dispose()
      val g2 = transientBuffer.createGraphics(); g2.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0f)); g2.fillRect(0,0,WIDTH,HEIGHT); g2.dispose()
      persistentG = persistentBuffer.createGraphics(); persistentG.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
      transientG = transientBuffer.createGraphics(); transientG.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
    }

    def initVertices(): Unit = {
      vertices.clear(); val cx = WIDTH/2.0; val cy = HEIGHT/2.0; val r = math.min(WIDTH, HEIGHT)*0.42
      var i = 0; while (i < nVertices) { val ang = 2*math.Pi*i/nVertices - math.Pi/2; val vx = cx + r*math.cos(ang); val vy = cy + r*math.sin(ang); vertices += Vertex(vx,vy, new Color((50+rand.nextInt(200))%256, (50+rand.nextInt(200))%256, (50+rand.nextInt(200))%256)); i += 1 }
      softmax = new SoftmaxBandit(nVertices); ucb = new UCB1Bandit(nVertices); ratioPerVertex = Array.fill(max(nVertices,10))(ratio)
      clearBuffers()
      persistentG.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1f)); persistentG.setStroke(new BasicStroke(1.2f))
      var j = 0; while (j < vertices.length) { val v = vertices(j); persistentG.setColor(new Color(v.color.getRed, v.color.getGreen, v.color.getBlue, 220)); persistentG.fillOval((v.x-5).toInt,(v.y-5).toInt,12,12); j += 1 }
      curX = rand.nextDouble()*WIDTH; curY = rand.nextDouble()*HEIGHT
      lastIndex = -1; totalPoints = 0; rewardSum = 0.0; stepsSinceReset = 0; frames = 0; itersPerTick = DEFAULT_ITERS_PER_TICK
    }

    // 局部密度 reward：在以 (cx,cy) 为中心，半径 r 的方形窗口内，空白像素比例（0..1）
    def localDensityReward(cx:Int, cy:Int, r:Int): Double = {
      if (cx - r < 0 || cy - r < 0 || cx + r >= WIDTH || cy + r >= HEIGHT) return 0.0
      var empty = 0; var total = 0
      var yy = cy - r
      while (yy <= cy + r) {
        var xx = cx - r
        while (xx <= cx + r) {
          if (persistentBuffer.getRGB(xx, yy) == bgColor.getRGB) empty += 1
          total += 1
          xx += 1
        }
        yy += 1
      }
      empty.toDouble / total.toDouble
    }

    def isBackgroundPixel(ix:Int, iy:Int): Boolean = {
      if (ix < 0 || ix >= WIDTH || iy < 0 || iy >= HEIGHT) return false
      persistentBuffer.getRGB(ix, iy) == bgColor.getRGB
    }

    override def actionPerformed(e: ActionEvent): Unit = {
      if (running) {
        if (autoRamp && itersPerTick < rampTarget) { frames += 1; if (frames % 15 == 0) itersPerTick = min(rampTarget, (itersPerTick * rampSpeed).toInt) }
        if (trailFade) { transientG.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 0.92f)); transientG.setColor(new Color(0,0,0,12)); transientG.fillRect(0,0,WIDTH,HEIGHT); transientG.setComposite(AlphaComposite.SrcOver) } else { transientG.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 0f)); transientG.fillRect(0,0,WIDTH,HEIGHT); transientG.setComposite(AlphaComposite.SrcOver) }

        var k = 0
        while (k < itersPerTick) {
          var chosen = -1
          var behaveClassic = false
          // HYBRID 模式下以概率 epsilon 执行 CLASSIC
          if (mode == Mode.HYBRID && rand.nextDouble() < epsilon) behaveClassic = true

          mode match {
            case Mode.CLASSIC => chosen = rand.nextInt(nVertices)
            case Mode.HYBRID => if (behaveClassic) chosen = rand.nextInt(nVertices) else {
              var attempt = 0; var cand = -1
              while (attempt < 12) { cand = softmax.sample(temperature); if (!(forbidSame && cand == lastIndex) && !(forbidNeighbors && lastIndex >= 0 && ((cand == (lastIndex+1)%nVertices) || (cand == (lastIndex-1+nVertices)%nVertices)))) attempt = 99 else attempt += 1 }
              if (cand < 0) cand = rand.nextInt(nVertices); chosen = cand
            }
            case Mode.AI_SOFTMAX =>
              var attempt = 0; var cand = -1
              while (attempt < 12) { cand = softmax.sample(temperature); if (!(forbidSame && cand == lastIndex) && !(forbidNeighbors && lastIndex >= 0 && ((cand == (lastIndex+1)%nVertices) || (cand == (lastIndex-1+nVertices)%nVertices)))) attempt = 99 else attempt += 1 }
              if (cand < 0) cand = rand.nextInt(nVertices); chosen = cand
            case Mode.AI_UCB1 =>
              var attempt = 0; var cand = -1
              while (attempt < 12) { cand = ucb.sample(); if (!(forbidSame && cand == lastIndex) && !(forbidNeighbors && lastIndex >= 0 && ((cand == (lastIndex+1)%nVertices) || (cand == (lastIndex-1+nVertices)%nVertices)))) attempt = 99 else attempt += 1 }
              if (cand < 0) cand = rand.nextInt(nVertices); chosen = cand
          }

          var tries = 0; var placed = false
          while (tries < maxFillAttempts && !placed) {
            val (tx,ty) = if (useMidpointMode) { var j = rand.nextInt(nVertices); while (j == chosen) j = rand.nextInt(nVertices); val v1 = vertices(chosen); val v2 = vertices(j); ((v1.x+v2.x)/2.0, (v1.y+v2.y)/2.0) } else { val v = vertices(chosen); (v.x, v.y) }
            val r = ratioPerVertex(chosen)
            val nx = curX + (tx - curX)*r; val ny = curY + (ty - curY)*r
            val ix = nx.toInt; val iy = ny.toInt

            if (mode == Mode.CLASSIC || (mode == Mode.HYBRID && behaveClassic)) {
              curX = nx; curY = ny
              val reward = localDensityReward(ix, iy, rewardRadius)
              mode match { case Mode.AI_SOFTMAX => softmax.update(chosen, reward, learningRate); case Mode.AI_UCB1 => ucb.update(chosen, reward); case _ => }
              rewardSum += reward; totalPoints += 1
              val col = colorMode match {
                case 0 => new Color(255,255,255,255)
                case 1 => new Color(vertices(chosen).color.getRed, vertices(chosen).color.getGreen, vertices(chosen).color.getBlue, 255)
                case 2 => new Color(((math.sin(totalPoints*0.0008 + chosen)*127)+128).toInt%256, ((math.cos(totalPoints*0.0006 + chosen)*127)+128).toInt%256, ((math.sin(totalPoints*0.0003 + chosen*2)*127)+128).toInt%256, 255)
                case _ => new Color(255,255,255,255)
              }
              if (ix >= 0 && ix < WIDTH && iy >= 0 && iy < HEIGHT) persistentBuffer.setRGB(ix, iy, col.getRGB)
              if (ix >= 0 && ix < WIDTH && iy >= 0 && iy < HEIGHT) transientBuffer.setRGB(ix, iy, latestDotColor.getRGB)
              lastIndex = chosen; placed = true
            } else {
              if (isBackgroundPixel(ix, iy)) {
                curX = nx; curY = ny
                val reward = localDensityReward(ix, iy, rewardRadius)
                mode match { case Mode.AI_SOFTMAX | Mode.HYBRID => softmax.update(chosen, reward, learningRate); case Mode.AI_UCB1 => ucb.update(chosen, reward); case _ => }
                rewardSum += reward; totalPoints += 1
                val col = colorMode match {
                  case 0 => new Color(255,255,255,255)
                  case 1 => new Color(vertices(chosen).color.getRed, vertices(chosen).color.getGreen, vertices(chosen).color.getBlue, 255)
                  case 2 => new Color(((math.sin(totalPoints*0.0008 + chosen)*127)+128).toInt%256, ((math.cos(totalPoints*0.0006 + chosen)*127)+128).toInt%256, ((math.sin(totalPoints*0.0003 + chosen*2)*127)+128).toInt%256, 255)
                  case _ => new Color(255,255,255,255)
                }
                persistentBuffer.setRGB(ix, iy, col.getRGB); transientBuffer.setRGB(ix, iy, latestDotColor.getRGB)
                lastIndex = chosen; placed = true
              } else {
                mode match { case Mode.AI_SOFTMAX | Mode.HYBRID => softmax.update(chosen, 0.0, learningRate*0.5); case Mode.AI_UCB1 => ucb.update(chosen, 0.0); case _ => }
                val smallStep = 0.25 * r
                curX = curX + (tx - curX) * smallStep
                curY = curY + (ty - curY) * smallStep
                lastIndex = chosen
              }
            }
            tries += 1
          }
          if (!placed) { curX = (curX + rand.nextDouble()*3 - 1.5); curY = (curY + rand.nextDouble()*3 - 1.5); totalPoints += 1 }
          if (totalPoints % 5000 == 0) try { softmax.normalize() } catch { case _: Throwable => () }
          k += 1
        }
        repaint()
      }
    }

    override def paintComponent(g: java.awt.Graphics): Unit = {
      val g2 = g.asInstanceOf[Graphics2D]
      g2.setColor(new Color(20,20,22)); g2.fillRect(0,0,getWidth,getHeight)
      g2.translate(offsetX, offsetY); g2.scale(scale, scale)
      g2.drawImage(persistentBuffer, 0, 0, null)
      g2.drawImage(transientBuffer, 0, 0, null)
      if (showHeatmap) {
        val hm = persistentBuffer; val w = hm.getWidth; val h = hm.getHeight
        val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB); val ig = img.createGraphics()
        var y = 0
        while (y < h) {
          var x = 0
          while (x < w) {
            val rgb = hm.getRGB(x,y)
            val r = (rgb>>16)&0xff; val gcol = (rgb>>8)&0xff; val b = rgb & 0xff
            val lum = (0.2126*r + 0.7152*gcol + 0.0722*b)/255.0
            val alpha = (lum*180).toInt
            val color = new Color(255,0,0, alpha)
            img.setRGB(x,y, color.getRGB)
            x += 1
          }
          y += 1
        }
        g2.drawImage(img, 0, 0, null); ig.dispose()
      }
      var i = 0; while (i < vertices.length) { val v = vertices(i); g2.setColor(new Color(v.color.getRed, v.color.getGreen, v.color.getBlue, 240)); g2.fillOval((v.x-6).toInt,(v.y-6).toInt,12,12); i += 1 }
      if (showLatestDot) { g2.setColor(latestDotColor); val px = (curX - 3).toInt; val py = (curY - 3).toInt; val size = math.max(1, (3.0/scale).toInt); g2.fillOval(px, py, size, size) }
      g2.scale(1.0/scale, 1.0/scale); g2.translate(-offsetX/scale, -offsetY/scale)
      val hx = getWidth - 56; val hy = 8
      g2.setColor(new Color(220,220,220,160)); g2.fillRoundRect(hx, hy, 44, 28, 6,6)
      g2.setColor(new Color(30,30,30,200)); g2.fillRect(hx+8, hy+6, 28, 3); g2.fillRect(hx+8, hy+12, 28, 3); g2.fillRect(hx+8, hy+18, 28, 3)
      g2.setColor(Color.WHITE); g2.setFont(new Font("Monospaced", Font.PLAIN, 12))
      val info = s"mode=${mode} verts=$nVertices ratio=${"%.3f".format(ratio)} itersTick=$itersPerTick pts=$totalPoints rewardAvg=${if (totalPoints>0) "%.4f".format(rewardSum/totalPoints) else "0.0000"} temp=${"%.2f".format(temperature)} lr=${"%.2f".format(learningRate)} scale=${"%.2f".format(scale)} eps=${"%.3f".format(epsilon)} rads=$rewardRadius"
      g2.drawString(info, 10, 18)
    }

    override def keyPressed(e: KeyEvent): Unit = {
      e.getKeyCode match {
        case KeyEvent.VK_SPACE => running = !running
        case KeyEvent.VK_R => initVertices()
        case KeyEvent.VK_S => saveSnapshot()
        case KeyEvent.VK_PLUS | KeyEvent.VK_EQUALS => scale = min(12.0, scale * 1.2); repaint()
        case KeyEvent.VK_MINUS | KeyEvent.VK_UNDERSCORE => scale = max(0.05, scale / 1.2); repaint()
        case KeyEvent.VK_0 => scale = 1.0; offsetX = 0; offsetY = 0; repaint()
        case KeyEvent.VK_1 => setVertices(3)
        case KeyEvent.VK_2 => setVertices(4)
        case KeyEvent.VK_3 => setVertices(5)
        case KeyEvent.VK_4 => setVertices(6)
        case KeyEvent.VK_5 => setVertices(7)
        case KeyEvent.VK_6 => setVertices(8)
        case _ =>
      }
    }
    override def keyReleased(e: KeyEvent): Unit = {}
    override def keyTyped(e: KeyEvent): Unit = {}

    // 以下为外部控件调用接口（中文）
    def setMode(m:Mode): Unit = { mode = m }
    def setVertices(n:Int): Unit = { nVertices = n; initVertices() }
    def setRatio(r:Double): Unit = { ratio = r; var i = 0; while (i < ratioPerVertex.length) { ratioPerVertex(i) = r; i += 1 } }
    def setItersPerTick(v:Int): Unit = { itersPerTick = v }
    def setForbidSame(b:Boolean): Unit = { forbidSame = b }
    def setForbidNeighbors(b:Boolean): Unit = { forbidNeighbors = b }
    def setMidpoint(b:Boolean): Unit = { useMidpointMode = b; initVertices() }
    def setTrail(b:Boolean): Unit = { trailFade = b }
    def setColorMode(m:Int): Unit = { colorMode = m }
    def setTemperature(t:Double): Unit = { temperature = t }
    def setLearningRate(lr:Double): Unit = { learningRate = lr }
    def setShowDot(b:Boolean): Unit = { showLatestDot = b }
    def setAutoRamp(b:Boolean): Unit = { autoRamp = b }
    def setMaxFillAttempts(n:Int): Unit = { maxFillAttempts = n }
    def setShowHeatmap(b:Boolean): Unit = { showHeatmap = b }
    def setEpsilon(v:Double): Unit = { epsilon = v }
    def setRewardRadius(r:Int): Unit = { rewardRadius = r }

    // 保存快照（PNG）
    def saveSnapshot(): Unit = {
      try {
        val outImg = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_ARGB)
        val g = outImg.createGraphics(); g.drawImage(persistentBuffer, 0, 0, null); g.drawImage(transientBuffer, 0, 0, null); g.dispose()
        val out = new File("chaos_snapshot_" + System.currentTimeMillis() + ".png")
        ImageIO.write(outImg, "png", out); println("保存 PNG 到 " + out.getAbsolutePath)
      } catch { case ex: Exception => ex.printStackTrace() }
    }
  } // 结束 ChaosPanel

  // 控件对话框（中文标签与 tooltip）
  def createControlDialog(parent: JFrame, panel: ChaosPanel): JDialog = {
    val dlg = new JDialog(parent, "控制面板", false)
    val ctrl = new JPanel(); ctrl.setLayout(new BoxLayout(ctrl, BoxLayout.Y_AXIS)); ctrl.setBorder(BorderFactory.createEmptyBorder(8,8,8,8)); ctrl.setBackground(new Color(40,40,44))
    def addRow(c: JComponent): Unit = { c.setAlignmentX(java.awt.Component.LEFT_ALIGNMENT); ctrl.add(c); ctrl.add(javax.swing.Box.createVerticalStrut(6)) }

    val modeLabel = new JLabel("模式："); modeLabel.setForeground(Color.WHITE); addRow(modeLabel)
    val g = new ButtonGroup()
    val rClassic = new JRadioButton("CLASSIC（经典）"); val rSoftmax = new JRadioButton("AI_SOFTMAX（Softmax）"); val rUCB = new JRadioButton("AI_UCB1（UCB1）"); val rHybrid = new JRadioButton("HYBRID（混合 ε-greedy）")
    rHybrid.setSelected(true); List(rClassic,rSoftmax,rUCB,rHybrid).foreach(rb => { rb.setBackground(new Color(40,40,44)); rb.setForeground(Color.WHITE); g.add(rb); addRow(rb) })
    rClassic.addActionListener((_: ActionEvent) => panel.setMode(Mode.CLASSIC))
    rSoftmax.addActionListener((_: ActionEvent) => panel.setMode(Mode.AI_SOFTMAX))
    rUCB.addActionListener((_: ActionEvent) => panel.setMode(Mode.AI_UCB1))
    rHybrid.addActionListener((_: ActionEvent) => panel.setMode(Mode.HYBRID))

    // Slider 与控件，均设中文 tooltip（含公式/说明）
    val ratioSlider = new JSlider(1,99,(DEFAULT_RATIO*100).toInt)
    ratioSlider.setToolTipText("移动比例 r：用于迭代公式 x_{n+1} = x_n + r*(v - x_n)。取值 0.01 ~ 0.99。")
    ratioSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setRatio(ratioSlider.getValue/100.0) }); addRow(new JLabel("比例 r：")); addRow(ratioSlider)

    val iterSlider = new JSlider(0,50000,DEFAULT_ITERS_PER_TICK)
    iterSlider.setToolTipText("每个渲染周期的迭代次数（影响收敛速度与可视效果）。")
    iterSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setItersPerTick(iterSlider.getValue) }); addRow(new JLabel("每周期迭代：")); addRow(iterSlider)

    val tempSlider = new JSlider(1,200,80)
    tempSlider.setToolTipText("Softmax 温度 T：在 Softmax 中 p_i ∝ exp(w_i / T)。T 越小越偏向 exploit，越大越随机。")
    tempSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setTemperature(tempSlider.getValue/100.0) }); addRow(new JLabel("温度 T：")); addRow(tempSlider)

    val lrSlider = new JSlider(1,100,25)
    lrSlider.setToolTipText("带权更新学习率 α：Softmax 更新 w_i ← w_i + α (reward - 0.5)。")
    lrSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setLearningRate(lrSlider.getValue/100.0) }); addRow(new JLabel("学习率 α：")); addRow(lrSlider)

    val maxTrySlider = new JSlider(1,50,12)
    maxTrySlider.setToolTipText("当 AI 选择的目标像素已被占用时，最多尝试寻找替代位置的次数（探索尝试）。")
    maxTrySlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setMaxFillAttempts(maxTrySlider.getValue) }); addRow(new JLabel("最大尝试次数：")); addRow(maxTrySlider)

    val epsSlider = new JSlider(0,100,4)
    epsSlider.setToolTipText("混合策略 ε（HYBRID）：以概率 ε 执行 CLASSIC（均匀采样），以 (1-ε) 执行 AI 策略。")
    epsSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setEpsilon(epsSlider.getValue/100.0) }); addRow(new JLabel("ε（混合概率）：")); addRow(epsSlider)

    val radSlider = new JSlider(1,12,4)
    radSlider.setToolTipText("局部密度半径 r：reward 通过在以候选像素为中心的 (2r+1)^2 窗口内计算空白像素比例得到。reward = 空白像素数 / 窗口总像素数。")
    radSlider.addChangeListener(new ChangeListener { def stateChanged(e: javax.swing.event.ChangeEvent): Unit = panel.setRewardRadius(radSlider.getValue) }); addRow(new JLabel("reward 半径 r：")); addRow(radSlider)

    val forbidSame = new JCheckBox("禁止连续选择相同顶点")
    forbidSame.setToolTipText("若勾选：若上一次选择的是顶点 i，则本次不能再选 i（避免重复）。")
    forbidSame.setSelected(true); forbidSame.addActionListener((_: ActionEvent) => panel.setForbidSame(forbidSame.isSelected)); addRow(forbidSame)

    val forbidNbr = new JCheckBox("禁止选择相邻顶点")
    forbidNbr.setToolTipText("若勾选：禁止选择与上一次选择顶点相邻的顶点（用于构造特定受限 chaos game）。")
    forbidNbr.setSelected(false); forbidNbr.addActionListener((_: ActionEvent) => panel.setForbidNeighbors(forbidNbr.isSelected)); addRow(forbidNbr)

    val midpoint = new JCheckBox("中点模式（选择两顶点的中点）")
    midpoint.setToolTipText("若勾选：目标不是单一顶点，而是随机选两顶点 v_i,v_j 的中点 m=(v_i+v_j)/2，迭代向 m 移动。")
    midpoint.setSelected(false); midpoint.addActionListener((_: ActionEvent) => panel.setMidpoint(midpoint.isSelected)); addRow(midpoint)

    val showDot = new JCheckBox("显示最新点高亮")
    showDot.setToolTipText("在 transient buffer 中高亮最近生成的像素，便于观察动态。")
    showDot.setSelected(true); showDot.addActionListener((_: ActionEvent) => panel.setShowDot(showDot.isSelected)); addRow(showDot)

    val trail = new JCheckBox("显示尾迹（渐隐）")
    trail.setToolTipText("是否对 transient buffer 每帧施加轻微覆盖以形成尾迹效果。")
    trail.setSelected(true); trail.addActionListener((_: ActionEvent) => panel.setTrail(trail.isSelected)); addRow(trail)

    val heat = new JCheckBox("热力图覆盖（显示密度）")
    heat.setToolTipText("热力图：红色显示已填充区域的密度（便于定位稀疏区域）。")
    heat.setSelected(false); heat.addActionListener((_: ActionEvent) => panel.setShowHeatmap(heat.isSelected)); addRow(heat)

    val autoRamp = new JCheckBox("自动递增迭代数（Auto ramp）")
    autoRamp.setToolTipText("自动逐步把 iters/tick 提升到目标（rampTarget），有助于先观察随机初态再加速收敛。")
    autoRamp.setSelected(true); autoRamp.addActionListener((_: ActionEvent) => panel.setAutoRamp(autoRamp.isSelected)); addRow(autoRamp)

    val colorBtn = new JButton("切换颜色模式")
    colorBtn.setToolTipText("颜色模式：0 白点；1 按顶点颜色；2 动态调色。")
    colorBtn.addActionListener((_: ActionEvent) => panel.setColorMode((panel.colorMode+1)%3)); addRow(colorBtn)

    val saveBtn = new JButton("保存 PNG")
    saveBtn.setToolTipText("把 persistent + transient 合并并导出为 PNG，用于后续分析或制作视频。")
    saveBtn.addActionListener((_: ActionEvent) => panel.saveSnapshot()); addRow(saveBtn)

    val closeBtn = new JButton("关闭")
    closeBtn.addActionListener((_: ActionEvent) => dlg.setVisible(false)); addRow(closeBtn)

    dlg.getContentPane.add(ctrl); dlg.pack(); dlg.setResizable(false); dlg.setLocationRelativeTo(parent)
    dlg
  }

  // 创建主窗口并绑定对话框（汉化）
  def createAndShowGUI(): Unit = {
    val frame = new JFrame("Chaos Game （画布）")
    frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE)
    val chaos = new ChaosPanel()
    frame.getContentPane.add(chaos)
    frame.pack(); frame.setResizable(true); frame.setLocationRelativeTo(null); frame.setVisible(true)
    val ctrlDlg = createControlDialog(frame, chaos)
    // 点击右上角汉堡切换控件窗口
    chaos.addMouseListener(new MouseAdapter {
      override def mouseClicked(e: MouseEvent): Unit = {
        if (e.getX >= chaos.getWidth-56 && e.getY >= 8 && e.getX <= chaos.getWidth-12 && e.getY <= 36) { ctrlDlg.setVisible(!ctrlDlg.isVisible) }
      }
    })
  }
} // 结束

object Runner { def main(args:Array[String]): Unit = SwingUtilities.invokeLater(() => AdvancedChaosGame.createAndShowGUI()) }
