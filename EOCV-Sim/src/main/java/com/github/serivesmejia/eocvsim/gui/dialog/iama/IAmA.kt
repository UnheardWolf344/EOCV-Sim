package com.github.serivesmejia.eocvsim.gui.dialog.iama

import com.github.serivesmejia.eocvsim.gui.EOCVSimIconLibrary
import com.github.serivesmejia.eocvsim.gui.Visualizer
import java.awt.Dimension
import java.awt.GridBagConstraints
import java.awt.GridBagLayout
import java.awt.GridLayout
import javax.swing.JButton
import javax.swing.JDialog
import javax.swing.JFrame
import javax.swing.JLabel
import javax.swing.JPanel

class IAmA(
    val parent: JFrame,
    val visualizer: Visualizer
) {

    val dialog = JDialog(parent)

    init {
        dialog.isModal = true
        dialog.title = "Welcome !"

        dialog.contentPane.layout = GridBagLayout()

        val text = """
            <b>Welcome to EOCV-Sim! We'll start with a walkthrough.</b><br>
            Please select the option that best describes you.<br><br>
            <b>I am..</b>
        """.trimIndent()

        dialog.contentPane.add(JLabel("<html><div style='text-align: center;'>$text</div></html>").apply {
            font = font.deriveFont(14f)
        }, GridBagConstraints().apply {
            gridx = 0
            gridy = 0
            weightx = 1.0
            weighty = 1.0
        })

        val buttonsPanel = JPanel().apply {
            layout = GridLayout(1, 3, 10, 10)
        }

        buttonsPanel.add(JButton(
            "<html><div style='text-align: center;'>A FIRST Robotics Team</div></html>",
            EOCVSimIconLibrary.icoFirstRobotics.scaleToFit(60, 60)
        ).apply {
            font = font.deriveFont(14f)
            horizontalTextPosition = JButton.CENTER
            verticalTextPosition = JButton.BOTTOM

            addActionListener {
                dialog.dispose()
                IAmAFirstRobotics(this@IAmA.parent, visualizer)
            }
        })

        buttonsPanel.add(JButton(
            "<html><div style='text-align: center;'>A General Public User</div></html>",
            EOCVSimIconLibrary.icoUser.scaleToFit(60, 60)
        ).apply {
            font = font.deriveFont(14f)
            horizontalTextPosition = JButton.CENTER
            verticalTextPosition = JButton.BOTTOM

            addActionListener {
                dialog.dispose()
                IAmAGeneralPublic(this@IAmA.parent, visualizer)
            }
        })

        buttonsPanel.add(JButton(
            "<html><div style='text-align: center;'>Specifically Interested<br>in PaperVision</div></html>",
            EOCVSimIconLibrary.icoPaperVision.scaleToFit(60, 60)
        ).apply {
            font = font.deriveFont(14f)
            horizontalTextPosition = JButton.CENTER
            verticalTextPosition = JButton.BOTTOM

            addActionListener {
                dialog.dispose()
                IAmAPaperVision(this@IAmA.parent, visualizer, specificallyInterested = true)
            }
        })

        dialog.contentPane.add(buttonsPanel, GridBagConstraints().apply {
            gridx = 0
            gridy = 1
            weightx = 1.0
            weighty = 1.0
        })

        dialog.size = Dimension(600, 250)
        dialog.isResizable = false

        dialog.defaultCloseOperation = JDialog.DISPOSE_ON_CLOSE
        dialog.setLocationRelativeTo(null)

        visualizer.eocvSim.config.flags["hasShownIamA"] = true
        dialog.isVisible = true
    }

}