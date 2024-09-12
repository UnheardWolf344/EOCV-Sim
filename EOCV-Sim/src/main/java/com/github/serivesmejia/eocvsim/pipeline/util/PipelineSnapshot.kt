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

package com.github.serivesmejia.eocvsim.pipeline.util

import com.github.serivesmejia.eocvsim.util.loggerForThis
import io.github.deltacv.eocvsim.virtualreflect.VirtualField
import io.github.deltacv.eocvsim.virtualreflect.VirtualReflectContext
import io.github.deltacv.eocvsim.virtualreflect.jvm.JvmVirtualReflectContext
import io.github.deltacv.eocvsim.virtualreflect.jvm.JvmVirtualReflection
import org.openftc.easyopencv.OpenCvPipeline
import java.util.*

class PipelineSnapshot(val virtualReflectContext: VirtualReflectContext, filter: ((VirtualField) -> Boolean)? = null) {

    val logger by loggerForThis()

    val holdingPipelineName = virtualReflectContext.simpleName

    val pipelineClass get() = (virtualReflectContext as JvmVirtualReflectContext).clazz

    val pipelineFieldValues: Map<VirtualField, Any?>

    init {
        val fieldValues = mutableMapOf<VirtualField, Any?>()

        for(field in virtualReflectContext.fields) {
            if(field.isFinal || field.isFinal)
                continue

            if(filter?.invoke(field) == false) continue

            fieldValues[field] = field.get()
        }

        pipelineFieldValues = fieldValues.toMap()

        logger.info("Taken snapshot of pipeline ${pipelineClass.name}")
    }

    fun transferTo(otherPipeline: OpenCvPipeline,
                   lastInitialPipelineSnapshot: PipelineSnapshot? = null) {
        if(pipelineClass.name != otherPipeline::class.java.name) return

        val changedList = if(lastInitialPipelineSnapshot != null)
            getChangedFieldsComparedTo(PipelineSnapshot(JvmVirtualReflection.contextOf(otherPipeline)), lastInitialPipelineSnapshot)
        else Collections.emptyList()

        fieldValuesLoop@
        for((field, value) in pipelineFieldValues) {
            for(changedField in changedList) {
                if(changedField.name == field.name && changedField.type == field.type) {
                    logger.trace(
                        "Skipping field ${field.name} since its value was changed in code, compared to the initial state of the pipeline"
                    )

                    continue@fieldValuesLoop
                }
            }

            try {
                field.set(value)
            } catch(e: Exception) {
                logger.trace(
                    "Failed to set field ${field.name} from snapshot of ${pipelineClass.name}. " +
                    "Retrying with by name lookup logic..."
                )

                try {
                    val byNameField = otherPipeline::class.java.getDeclaredField(field.name)
                    byNameField.set(otherPipeline, value)
                } catch(e: Exception) {
                    logger.warn(
                        "Definitely failed to set field ${field.name} from snapshot of ${pipelineClass.name}. Did the source code change?",
                        e
                    )
                }
            }
        }
    }

    fun getField(name: String): Pair<VirtualField, Any?>? {
        for((field, value) in pipelineFieldValues) {
            if(field.name == name) {
                return Pair(field, value)
            }
        }

        return null
    }

    private fun getChangedFieldsComparedTo(
        pipelineSnapshotA: PipelineSnapshot,
        pipelineSnapshotB: PipelineSnapshot
    ): List<VirtualField> = pipelineSnapshotA.run {
        if(holdingPipelineName != pipelineSnapshotB.holdingPipelineName && pipelineClass != pipelineSnapshotB.pipelineClass)
            return Collections.emptyList()

        val changedList = mutableListOf<VirtualField>()

        for((field, value) in pipelineFieldValues) {
            val (otherField, otherValue) = pipelineSnapshotB.getField(field.name) ?: continue
            if (field.type != otherField.type) continue

            if(otherValue != value) {
                changedList.add(field)
            }
        }

        return changedList
    }

}
