/*
 * Copyright (c) 2021 Sebastian Erives
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package com.github.serivesmejia.eocvsim.util.compiler

import com.github.serivesmejia.eocvsim.util.loggerOf
import org.eclipse.jdt.internal.compiler.tool.EclipseCompiler
import javax.tools.JavaCompiler
import javax.tools.ToolProvider

private val logger by loggerOf("CompilerProvider")

val compiler by lazy {
    val toolProviderCompiler = try {
        ToolProvider.getSystemJavaCompiler()
    } catch(e: Exception) {
        logger.warn("ToolProvider threw an exception on getSystemJavaCompiler()", e)
        null
    }

    try {
        if (toolProviderCompiler == null) {
            Compiler("Eclipse", EclipseCompiler())
        } else {
            Compiler("JDK", toolProviderCompiler)
        }
    } catch(e: Exception) {
        logger.warn("Unexpected exception while providing a java compiler", e)
        null
    }
}

data class Compiler(val name: String, val javaCompiler: JavaCompiler)