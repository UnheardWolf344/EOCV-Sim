package com.github.serivesmejia.eocvsim.util.exception.handling

import com.github.serivesmejia.eocvsim.currentMainThread
import com.github.serivesmejia.eocvsim.util.JavaProcess
import com.github.serivesmejia.eocvsim.util.SysUtil
import com.github.serivesmejia.eocvsim.util.loggerForThis
import java.lang.Thread.sleep
import javax.swing.SwingUtilities
import kotlin.system.exitProcess

class EOCVSimUncaughtExceptionHandler private constructor() : Thread.UncaughtExceptionHandler {

    companion object {
        const val MAX_UNCAUGHT_EXCEPTIONS_BEFORE_CRASH = 3

        @JvmStatic fun register() {
            Thread.setDefaultUncaughtExceptionHandler(EOCVSimUncaughtExceptionHandler())
        }
    }

    val logger by loggerForThis()

    private var uncaughtExceptionsCount = 0

    override fun uncaughtException(t: Thread, e: Throwable) {
        //we don't want the whole app to crash on a simple interrupted exception right?
        if(e is InterruptedException) {
            logger.warn("Uncaught InterruptedException thrown in Thread ${t.name}, it will be interrupted", e)
            t.interrupt()
            return
        }

        uncaughtExceptionsCount++

        logger.error("Uncaught exception thrown in \"${t.name}\" thread", e)

        //Exit if uncaught exception happened in the main thread
        //since we would be basically in a deadlock state if that happened
        //or if we have a lotta uncaught exceptions.
        if(t == currentMainThread || SwingUtilities.isEventDispatchThread() || e !is Exception || uncaughtExceptionsCount > MAX_UNCAUGHT_EXCEPTIONS_BEFORE_CRASH) {
            val file = CrashReport(e).saveCrashReport()

            logger.warn("If this error persists, open an issue on EOCV-Sim's GitHub attaching the crash report file.")
            logger.warn("The application will exit now (exit code 1)")

            Thread {
                JavaProcess.killSubprocessesOnExit = false
                JavaProcess.exec(CrashReportOutputMain::class.java, listOf(), listOf("-p=${file.absolutePath}"))
            }.start()

            sleep(3000) // wait enough for the subprocess to start

            exitProcess(1)
        } else {
            CrashReport(e).saveCrashReport("lasterror-eocvsim")

            //if not, eocv sim might still be working (i.e a crash from a MatPoster thread)
            //so we might not need to exit in this point, but we'll need to send a warning
            //to the user
            logger.warn("If this error persists, open an issue on EOCV-Sim's GitHub.")
            logger.warn("The application might not work as expected from this point")
        }
    }

}